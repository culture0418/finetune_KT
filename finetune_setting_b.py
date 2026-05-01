"""
Setting B: RoBERTa 訓練（in-memory 移除 [學生表現] 監督訊號）
========================================================
診斷 Setting A（含 [學生表現]）99.x% accuracy 的 leakage 問題之 ablation 實驗。

設計：
- 不修改 datasets/，純 in-memory 從 Short_Answer_Log 剝除 ［學生表現］：xxx 行
- 訓練流程其他部分完全沿用 finetune_bert.py 的設定
  （Optuna 15 trials + 完整訓練 + 加權 CE + 早停）
- 完訓後自動對 test set inference，方便即時比較 Setting A vs Setting B

執行：
    python finetune_setting_b.py

輸出：
    results/roberta-chinese_setting_b_<timestamp>/
        final_model/                  # 最佳 epoch 之 model 權重
        training_metrics_summary.csv  # 各 epoch val 指標
        test_metrics.json             # test set 指標（與 Setting A 對照）
        test_predictions.csv          # 逐筆預測
        training_metrics_visualization.png
        per_class_metrics_visualization.png
        confusion_matrix_heatmap.png
"""

import json
import os
import re
from datetime import datetime

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import TrainingArguments

from finetune_bert import (
    KTDataProcessor,
    KTFinetuner,
    HyperparameterSearcher,
    ensure_dir_exists,
)


# 規則式剝除 [學生表現]：xxx 整行（含結尾換行）
PERF_LINE_PATTERN = re.compile(r'［學生表現］：[^\n]*\n?')

LABEL_LIST = ["待加強", "尚可", "精熟"]


class KTDataProcessorNoPerf(KTDataProcessor):
    """KTDataProcessor 子類：載入後從 Short_Answer_Log 剝除 ［學生表現］：xxx 整行。"""

    def _load_split_csv(self, name: str):
        df = super()._load_split_csv(name)
        if 'Short_Answer_Log' in df.columns:
            df['Short_Answer_Log'] = df['Short_Answer_Log'].apply(
                lambda s: PERF_LINE_PATTERN.sub('', s)
            )
            remaining = df['Short_Answer_Log'].str.contains('［學生表現］').sum()
            print(f"  [{name}] 剝除 [學生表現] 後仍殘留筆數: {remaining}")
        return df


def evaluate_on_test(model, tokenizer, test_df, device):
    """對 test set inference，回傳指標 dict 與逐筆預測 list。"""
    model.eval()
    label_map = {"待加強": 0, "尚可": 1, "精熟": 2}
    id2label = {v: k for k, v in label_map.items()}

    preds = []
    with torch.no_grad():
        for _, row in test_df.iterrows():
            text = f"{row['chapter']} {row['section']} {row['Short_Answer_Log']}"
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=512,
            ).to(device)
            logits = model(**inputs).logits
            preds.append(id2label[logits.argmax(dim=-1).item()])

    y_true = test_df['Mastery_Label'].tolist()
    acc = accuracy_score(y_true, preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, preds, labels=LABEL_LIST, average='macro', zero_division=0
    )
    p_per, r_per, f1_per, support = precision_recall_fscore_support(
        y_true, preds, labels=LABEL_LIST, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, preds, labels=LABEL_LIST)

    metrics = {
        "accuracy": float(acc),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "per_class": {
            lbl: {
                "precision": float(p_per[i]),
                "recall": float(r_per[i]),
                "f1": float(f1_per[i]),
                "support": int(support[i]),
            }
            for i, lbl in enumerate(LABEL_LIST)
        },
        "confusion_matrix": {
            "labels": LABEL_LIST,
            "matrix": cm.tolist(),
        },
    }
    return metrics, preds


def main():
    SPLITS_DIR = "datasets/splits/0227"
    N_TRIALS = 15
    FULL_TRAIN_EPOCHS = 15

    print("\n" + "=" * 80)
    print("🔬 Setting B: RoBERTa 訓練（in-memory 移除 [學生表現]）")
    print("=" * 80)

    roberta_model_name = "hfl/chinese-roberta-wwm-ext"
    roberta_output_base = "./optuna_results/roberta_setting_b"

    # ===== 1. Optuna 超參數搜索 =====
    print(f"\n>>> Optuna 超參數搜索（{N_TRIALS} trials）\n")

    searcher = HyperparameterSearcher(
        model_class=KTFinetuner,
        data_processor_class=KTDataProcessorNoPerf,
    )

    study = searcher.run_search(
        splits_dir=SPLITS_DIR,
        model_name=roberta_model_name,
        output_base_dir=roberta_output_base,
        n_trials=N_TRIALS,
        study_name="roberta_setting_b_hp_search",
    )

    best = study.best_params
    print(f"\n>>> Optuna best trial: macro_f1={study.best_value:.4f}")
    print(f"    params: {best}")

    # ===== 2. 完整訓練（沿用 RoBERTa 保守調整：lr × 0.5、warmup + 100） =====
    print("\n" + "=" * 80)
    print(f"🚀 完整訓練 RoBERTa (Setting B, {FULL_TRAIN_EPOCHS} epochs)")
    print("=" * 80 + "\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./results/roberta-chinese_setting_b_{timestamp}"
    model_path = f"{output_dir}/final_model"
    ensure_dir_exists(output_dir)
    ensure_dir_exists(model_path)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=FULL_TRAIN_EPOCHS,
        per_device_train_batch_size=best.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=4,
        learning_rate=best.get("learning_rate", 3e-5) * 0.5,
        warmup_steps=best.get("warmup_steps", 100) + 100,
        weight_decay=best.get("weight_decay", 0.01),
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",
        seed=42,
        data_seed=42,
    )

    dp = KTDataProcessorNoPerf(splits_dir=SPLITS_DIR)
    dp.prepare_data()

    train_df, _ = dp.get_dataframes()
    label_counts = train_df['labels'].value_counts().sort_index()
    total = len(train_df)
    class_weights = torch.tensor(
        [
            total / (3 * label_counts[0]),
            total / (3 * label_counts[1]),
            total / (3 * label_counts[2]),
        ],
        dtype=torch.float32,
    )
    print(f"\n📋 類別權重: 待加強={class_weights[0]:.2f}, "
          f"尚可={class_weights[1]:.2f}, 精熟={class_weights[2]:.2f}")

    finetuner = KTFinetuner(
        model_name=roberta_model_name,
        data_processor=dp,
        training_args=training_args,
        max_token_len=512,
        class_weights=class_weights,
        early_stopping_patience=3,
    )
    finetuner.run_finetuning()
    finetuner.save_model(save_path=model_path)

    # ===== 3. Test set 評測 =====
    print("\n" + "=" * 80)
    print("📊 Test Set 評測")
    print("=" * 80)

    test_metrics, test_preds = evaluate_on_test(
        finetuner.model, finetuner.tokenizer, dp.test_df, finetuner.device
    )

    # 存指標
    metrics_path = os.path.join(output_dir, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    # 存逐筆預測
    pred_df = dp.test_df.copy()
    pred_df['predicted'] = test_preds
    pred_path = os.path.join(output_dir, "test_predictions.csv")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    # ===== 4. 摘要輸出 =====
    print(f"\n>>> Test 結果:")
    print(f"   Accuracy:        {test_metrics['accuracy']:.4f}")
    print(f"   Macro Precision: {test_metrics['macro_precision']:.4f}")
    print(f"   Macro Recall:    {test_metrics['macro_recall']:.4f}")
    print(f"   Macro F1:        {test_metrics['macro_f1']:.4f}")
    for lbl in LABEL_LIST:
        m = test_metrics['per_class'][lbl]
        print(f"   {lbl}: P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} (n={m['support']})")

    print(f"\n>>> 對照 Setting A（含 [學生表現]）:")
    print(f"   Setting A test macro_f1 = 0.983 ± 0.016 (n=5 seeds)")
    print(f"   Setting B test macro_f1 = {test_metrics['macro_f1']:.4f} (single seed=42)")

    print(f"\n✅ 模型存於: {model_path}")
    print(f"   指標: {metrics_path}")
    print(f"   預測: {pred_path}")


if __name__ == "__main__":
    main()
