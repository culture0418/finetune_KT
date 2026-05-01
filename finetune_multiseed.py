"""
Multi-seed RoBERTa 訓練 + Test Set 評測
========================================
跑 N 個 random seed 的 RoBERTa 訓練（沿用 datasets/splits/0227 + 第一次 Optuna 跑出的 best params），
每個 seed 訓練完後在 test set 上 inference，最後彙整 mean ± std。

不重跑 Optuna（節省時間），不重跑 LLM（test set 沒變）。

執行：
    python finetune_multiseed.py
輸出：
    results/multiseed_<timestamp>/
        seed_results.csv      # 每個 seed 的 test 指標
        seed_summary.csv      # mean ± std 總表
        seed_<n>/             # 各 seed 的訓練 log（會自動清理 checkpoint）
"""

import os
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import TrainingArguments

from finetune_bert import (
    KTDataProcessor,
    KTFinetuner,
    ensure_dir_exists,
)


# ===== 設定 =====
SEEDS = [42, 7, 123, 2024, 999]
SPLITS_DIR = "datasets/splits/0227"
ROBERTA_MODEL = "hfl/chinese-roberta-wwm-ext"

# 從第一次 Optuna best trial (#3, optuna_results/roberta/study_results.csv) 直接取
OPTUNA_BEST = {
    "learning_rate": 3.887891717892839e-05,
    "warmup_steps": 50,
    "weight_decay": 0.021456057466280057,
    "per_device_train_batch_size": 8,
}

# 完整訓練的保守調整（沿用 finetune_bert.py 的策略）
FULL_TRAIN_LR_MULT = 0.5
FULL_TRAIN_WARMUP_OFFSET = 100
FULL_TRAIN_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 3

# 標籤
LABEL_MAP = {"待加強": 0, "尚可": 1, "精熟": 2}
LABEL_LIST = list(LABEL_MAP.keys())
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}


def evaluate_on_test(model, tokenizer, test_df, device):
    """對 test set 做 inference 並回傳指標 dict。"""
    model.eval()
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
            pred_id = logits.argmax(dim=-1).item()
            preds.append(ID2LABEL[pred_id])

    y_true = test_df["Mastery_Label"].tolist()
    acc = accuracy_score(y_true, preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, preds, labels=LABEL_LIST, average="macro", zero_division=0
    )
    _, _, f1_per, _ = precision_recall_fscore_support(
        y_true, preds, labels=LABEL_LIST, average=None, zero_division=0
    )
    return {
        "accuracy": acc,
        "macro_precision": p_macro,
        "macro_recall": r_macro,
        "macro_f1": f1_macro,
        "待加強_f1": f1_per[0],
        "尚可_f1": f1_per[1],
        "精熟_f1": f1_per[2],
    }, preds


def cleanup_checkpoints(output_dir):
    """訓練結束後刪除 checkpoint-* 目錄釋放磁碟（每個 ~390MB）。"""
    if not os.path.isdir(output_dir):
        return
    for name in os.listdir(output_dir):
        full = os.path.join(output_dir, name)
        if os.path.isdir(full) and name.startswith("checkpoint-"):
            shutil.rmtree(full)


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = f"./results/multiseed_{timestamp}"
    ensure_dir_exists(output_root)

    print("=" * 80)
    print(f"🔬 RoBERTa Multi-Seed 訓練 (n={len(SEEDS)})  splits={SPLITS_DIR}")
    print("=" * 80)
    print(f"  超參數 (沿用 Optuna best params):")
    print(f"    learning_rate = {OPTUNA_BEST['learning_rate']:.4e} × {FULL_TRAIN_LR_MULT}")
    print(f"    warmup_steps  = {OPTUNA_BEST['warmup_steps']} + {FULL_TRAIN_WARMUP_OFFSET}")
    print(f"    weight_decay  = {OPTUNA_BEST['weight_decay']:.4f}")
    print(f"    batch_size    = {OPTUNA_BEST['per_device_train_batch_size']}")
    print(f"    epochs        = {FULL_TRAIN_EPOCHS} (early stop patience={EARLY_STOPPING_PATIENCE})")
    print(f"  Seeds: {SEEDS}")

    all_rows = []
    all_predictions = {}

    for seed in SEEDS:
        print(f"\n{'='*80}\n>>> SEED {seed}\n{'='*80}")
        seed_dir = f"{output_root}/seed_{seed}"
        ensure_dir_exists(seed_dir)

        # 1. 載入資料（每個 seed 都重新建一個 processor，確保乾淨）
        dp = KTDataProcessor(splits_dir=SPLITS_DIR)
        dp.prepare_data()

        train_df, _ = dp.get_dataframes()
        label_counts = train_df["labels"].value_counts().sort_index()
        total = len(train_df)
        class_weights = torch.tensor(
            [
                total / (3 * label_counts[0]),
                total / (3 * label_counts[1]),
                total / (3 * label_counts[2]),
            ],
            dtype=torch.float32,
        )

        # 2. 訓練設定
        training_args = TrainingArguments(
            output_dir=seed_dir,
            num_train_epochs=FULL_TRAIN_EPOCHS,
            per_device_train_batch_size=OPTUNA_BEST["per_device_train_batch_size"],
            per_device_eval_batch_size=4,
            learning_rate=OPTUNA_BEST["learning_rate"] * FULL_TRAIN_LR_MULT,
            warmup_steps=OPTUNA_BEST["warmup_steps"] + FULL_TRAIN_WARMUP_OFFSET,
            weight_decay=OPTUNA_BEST["weight_decay"],
            logging_dir=f"{seed_dir}/logs",
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            report_to="none",
            seed=seed,
            data_seed=seed,
            disable_tqdm=True,
        )

        # 3. 訓練
        finetuner = KTFinetuner(
            model_name=ROBERTA_MODEL,
            data_processor=dp,
            training_args=training_args,
            max_token_len=512,
            class_weights=class_weights,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
        )
        finetuner.run_finetuning()

        # 4. Test set 評測
        metrics, preds = evaluate_on_test(
            finetuner.model, finetuner.tokenizer, dp.test_df, finetuner.device
        )
        metrics["seed"] = seed
        all_rows.append(metrics)
        all_predictions[seed] = preds

        n_correct = int(metrics["accuracy"] * len(dp.test_df))
        print(
            f"\n>>> SEED {seed} 結果:  "
            f"acc={metrics['accuracy']:.4f} ({n_correct}/{len(dp.test_df)})  "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )

        # 5. 清理 GPU + checkpoint
        del finetuner
        torch.cuda.empty_cache()
        cleanup_checkpoints(seed_dir)

    # ===== 彙整 =====
    df = pd.DataFrame(all_rows)[
        ["seed", "accuracy", "macro_precision", "macro_recall", "macro_f1",
         "待加強_f1", "尚可_f1", "精熟_f1"]
    ]
    df.to_csv(f"{output_root}/seed_results.csv", index=False, encoding="utf-8-sig")

    metric_cols = ["accuracy", "macro_precision", "macro_recall", "macro_f1",
                   "待加強_f1", "尚可_f1", "精熟_f1"]
    summary = pd.DataFrame({
        "metric": metric_cols,
        "mean": [df[m].mean() for m in metric_cols],
        "std": [df[m].std(ddof=1) for m in metric_cols],
        "min": [df[m].min() for m in metric_cols],
        "max": [df[m].max() for m in metric_cols],
    })
    summary.to_csv(f"{output_root}/seed_summary.csv", index=False, encoding="utf-8-sig")

    # 各 seed 預測也存一份 (debugging 用)
    pred_df = pd.DataFrame(all_predictions)
    pred_df.insert(0, "true", dp.test_df["Mastery_Label"].values)
    pred_df.to_csv(f"{output_root}/per_seed_predictions.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("📊 Multi-Seed 結果總表")
    print("=" * 80)
    print(df.to_string(index=False))

    print("\n" + "=" * 80)
    print(f"📊 Mean ± Std (n={len(SEEDS)})")
    print("=" * 80)
    for m in metric_cols:
        print(f"  {m:18s}: {df[m].mean():.4f} ± {df[m].std(ddof=1):.4f}  "
              f"(min={df[m].min():.4f}, max={df[m].max():.4f})")

    print(f"\n✅ 結果輸出至: {output_root}/")


if __name__ == "__main__":
    main()
