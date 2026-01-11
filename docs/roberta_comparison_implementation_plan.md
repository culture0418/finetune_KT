> **Git Branch**: `feature/add-roberta-comparison`  
> **規劃日期**: 2026-01-10

# 添加 RoBERTa 模型比較 - 簡化實作計畫

## 實作策略

**核心理念**：最小改動原則 - 在現有代碼基礎上添加 RoBERTa 支持，無需大規模重構。

### 為什麼選擇簡化方案？

1. ✅ **快速實驗**：您想盡快完成訓練進入 transfer 開發
2. ✅ **風險最小**：避免大規模重構可能引入的 bug
3. ✅ **向後兼容**：現有 BERT 訓練流程完全不受影響
4. ✅ **易於維護**：代碼改動集中在關鍵位置

---

## 確認的配置

- ✅ **資料集**: `finetune_dataset_1132_v2.csv` (725 筆，3 個等級)
- ✅ **BERT 模型**: `bert-base-chinese` + Optuna 搜索 (15 trials)
- ✅ **RoBERTa 模型**: `hfl/chinese-roberta-wwm-ext` + Optuna 搜索 (15 trials)
- ✅ **比較報告**: CSV 格式，保存至 `results/model_comparison_report.csv`
- ✅ **模型保存**: 兩個模型的最佳版本都要保存
  - 格式: `results/{model_name}_{YYYYMMDD_HHMMSS}`
  - 例如: `results/bert-base-chinese_20260110_130000`

---

## 關鍵疑問說明：資料集切分與比較公平性

針對您提到的 **「8:2 切分是否會導致模型看到全部資料，影響比較公平性？」**

**解答**：不會，這是一個標準且公平的比較方式，原因如下：

1.  **嚴格隔離 (Strict Separation)**：
    - 模型在訓練過程（Backpropagation/梯度更新）中**只會使用 80% 的訓練資料**來學習。
    - 另外 20% 的驗證資料僅用於「考試」（計算分數），模型**絕對不會**拿這 20% 來修改自己的參數。就像學生考試時看到了考卷題目，但不能把答案抄回課本裡一樣。

2.  **固定隨機與公平起跑點 (Consistent Split)**：
    - 我們在程式碼中強制設定了 `random_state=42`。
    - **結果**：BERT 和 RoBERTa 會分到**完全一模一樣**的訓練集（那特定的 80%）和**完全一模一樣**的驗證集（那特定的 20%）。
    - 因為大家考的是同一份考卷（驗證集），且都沒看過考卷內容，所以比較結果是**準確且公平**的。

---

## Proposed Changes

### [MODIFY] [finetune_bert.py](file:///home/culture/finetune_KT/finetune_bert.py)

#### 改動 1: 添加 RoBERTa 導入 (第 13-18 行附近)

在現有的 import 語句中添加：

```python
from transformers import (
    BertConfig,
    BertTokenizer,
    BertForSequenceClassification,
    RobertaConfig,          # 新增
    RobertaTokenizer,       # 新增（備用，中文 RoBERTa 通常用 BertTokenizer）
    RobertaForSequenceClassification,  # 新增
    TrainingArguments,
    Trainer
)
```

**位置**: 第 13-18 行  
**改動行數**: 3 行

---

#### 改動 2: 添加模型配置字典 (第 22 行之後)

在所有 import 之後，類別定義之前添加：

```python
# ========================================
# 模型配置字典
# ========================================
MODEL_CONFIGS = {
    "bert-base-chinese": {
        "config_class": BertConfig,
        "tokenizer_class": BertTokenizer,
        "model_class": BertForSequenceClassification,
        "max_token_len": 512,
        "description": "BERT Base Chinese"
    },
    "hfl/chinese-roberta-wwm-ext": {
        "config_class": RobertaConfig,
        "tokenizer_class": BertTokenizer,  # 中文 RoBERTa 使用 BertTokenizer
        "model_class": RobertaForSequenceClassification,
        "max_token_len": 512,
        "description": "RoBERTa WWM Chinese (Whole Word Masking)"
    }
}
```

**位置**: 第 22 行之後  
**改動行數**: 20 行（新增）

---

#### 改動 3: 重命名並通用化訓練類別 (第 607-643 行)

**將 `BertKTFinetuner` 重命名為 `KTFinetuner`** 並修改 `__init__` 方法：

```python
class KTFinetuner:  # 原本是 BertKTFinetuner
    """
    功用：通用的知識追蹤模型訓練類別。
    支援 BERT、RoBERTa 等多種 Transformer 模型。
    """
    def __init__(self, model_name: str, data_processor: KTDataProcessor, 
                 training_args: TrainingArguments, max_token_len: int = 512):
        """
        Args:
            model_name (str): 模型名稱，必須在 MODEL_CONFIGS 中定義
            data_processor (KTDataProcessor): 已經準備好資料的 DataProcessor 物件
            training_args (TrainingArguments): Hugging Face 的訓練參數
            max_token_len (int): 最大 Token 長度，預設 512
        """
        self.model_name = model_name
        self.processor = data_processor
        self.training_args = training_args
        self.max_token_len = max_token_len
        
        # 檢查模型是否在配置字典中
        if model_name not in MODEL_CONFIGS:
            raise ValueError(
                f"模型 '{model_name}' 不在支援列表中。\n"
                f"支援的模型: {list(MODEL_CONFIGS.keys())}"
            )
        
        model_config = MODEL_CONFIGS[model_name]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"模型將在 {self.device} 上運行")
        print(f"使用模型: {model_config['description']}")

        # 1. 載入 Tokenizer（使用配置字典中的類別）
        tokenizer_class = model_config["tokenizer_class"]
        self.tokenizer = tokenizer_class.from_pretrained(self.model_name)

        # 2. 載入模型（使用配置字典中的類別）
        config_class = model_config["config_class"]
        model_class = model_config["model_class"]
        
        config = config_class.from_pretrained(
            self.model_name,
            num_labels=self.processor.num_labels,
            id2label=self.processor.id2label,
            label2id=self.processor.label_map
        )
        self.model = model_class.from_pretrained(
            self.model_name,
            config=config
        ).to(self.device)
        
        self.trainer = None
```

**位置**: 第 607-643 行  
**改動行數**: 約 45 行（主要是 `__init__` 方法的邏輯改變）

---

#### 改動 4: 更新 HyperparameterSearcher 引用 (第 866-871 行)

將所有 `BertKTFinetuner` 引用改為 `KTFinetuner`：

```python
# 在 objective 函數中 (第 866 行附近)
finetuner = KTFinetuner(  # 原本是 BertKTFinetuner
    model_name=model_name,
    data_processor=data_processor,
    training_args=training_args,
    max_token_len=512
)
```

**位置**: 第 866-871 行  
**改動行數**: 1 行（類別名稱）

---

#### 改動 5: 添加比較報告生成函數 (第 953 行之後，`ensure_dir_exists` 函數之後)

```python
def generate_comparison_report(bert_results: dict, roberta_results: dict, output_path: str):
    """
    生成 BERT vs RoBERTa 比較報告並保存為 CSV
    
    Args:
        bert_results: BERT 訓練結果字典
        roberta_results: RoBERTa 訓練結果字典
        output_path: CSV 保存路徑
    """
    import pandas as pd
    from datetime import datetime
    
    print(f"\n{'='*80}")
    print("📊 生成模型比較報告")
    print(f"{'='*80}\n")
    
    # 準備比較數據
    comparison_data = {
        "模型名稱": ["BERT (bert-base-chinese)", "RoBERTa (hfl/chinese-roberta-wwm-ext)"],
        "Macro F1-Score": [
            bert_results.get("eval_macro_f1", 0),
            roberta_results.get("eval_macro_f1", 0)
        ],
        "整體準確率": [
            bert_results.get("eval_accuracy", 0),
            roberta_results.get("eval_accuracy", 0)
        ],
        "待加強_F1": [
            bert_results.get("eval_待加強_f1", 0),
            roberta_results.get("eval_待加強_f1", 0)
        ],
        "尚可_F1": [
            bert_results.get("eval_尚可_f1", 0),
            roberta_results.get("eval_尚可_f1", 0)
        ],
        "精熟_F1": [
            bert_results.get("eval_精熟_f1", 0),
            roberta_results.get("eval_精熟_f1", 0)
        ],
        "模型路徑": [
            bert_results.get("model_path", "N/A"),
            roberta_results.get("model_path", "N/A")
        ],
        "訓練時間戳": [
            bert_results.get("timestamp", "N/A"),
            roberta_results.get("timestamp", "N/A")
        ]
    }
    
    df = pd.DataFrame(comparison_data)
    
    # 保存 CSV
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ 比較報告已保存至: {output_path}\n")
    
    # 打印到終端
    print("="*80)
    print("📊 BERT vs RoBERTa 性能比較")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    # 判斷最佳模型
    bert_f1 = bert_results.get("eval_macro_f1", 0)
    roberta_f1 = roberta_results.get("eval_macro_f1", 0)
    
    if roberta_f1 > bert_f1:
        improvement = ((roberta_f1 - bert_f1) / bert_f1) * 100
        print(f"\n🏆 最佳模型: RoBERTa")
        print(f"📈 性能提升: {improvement:.2f}%")
        print(f"   Macro-F1: {bert_f1:.4f} → {roberta_f1:.4f}")
    elif bert_f1 > roberta_f1:
        improvement = ((bert_f1 - roberta_f1) / roberta_f1) * 100
        print(f"\n🏆 最佳模型: BERT")
        print(f"📈 性能優勢: {improvement:.2f}%")
        print(f"   Macro-F1: {roberta_f1:.4f} → {bert_f1:.4f}")
    else:
        print(f"\n⚖️  兩個模型性能相當")
        print(f"   Macro-F1: {bert_f1:.4f}")
    
    print("="*80 + "\n")
    
    return df
```

**位置**: 第 953 行之後  
**改動行數**: 85 行（新增函數）

---

#### 改動 6: 修改主程序支持兩模型訓練 (第 962-1145 行)

**完全重寫 `if __name__ == "__main__":` 區塊**：

```python
if __name__ == "__main__":
    
    # ========================================
    # 🎯 雙模型比較實驗配置
    # ========================================
    
    print("\n" + "="*80)
    print("🔬 BERT vs RoBERTa 知識追蹤模型比較實驗")
    print("="*80 + "\n")
    
    # 共用配置
    DATASET_PATH = "datasets/finetune_dataset_1132_v2.csv"
    N_TRIALS = 15  # Optuna 搜索次數
    FULL_TRAIN_EPOCHS = 60  # 使用最佳參數後的完整訓練輪數
    
    # 記錄訓練結果
    training_results = {}
    
    # ========================================
    # 實驗 1: BERT 模型
    # ========================================
    
    print("\n" + "="*80)
    print("🔍 實驗 1/2: BERT 模型 (bert-base-chinese)")
    print("="*80 + "\n")
    
    bert_model_name = "bert-base-chinese"
    bert_output_base = "./optuna_results/bert"
    
    # BERT Optuna 搜索
    print(f"開始 BERT 超參數搜索 ({N_TRIALS} trials)...\n")
    
    bert_searcher = HyperparameterSearcher(
        model_class=KTFinetuner,
        data_processor_class=KTDataProcessor
    )
    
    bert_study = bert_searcher.run_search(
        csv_path=DATASET_PATH,
        model_name=bert_model_name,
        output_base_dir=bert_output_base,
        n_trials=N_TRIALS,
        study_name="bert_hp_search"
    )
    
    # BERT 完整訓練
    print("\n" + "="*80)
    print(f"🚀 使用最佳參數訓練 BERT ({FULL_TRAIN_EPOCHS} epochs)")
    print("="*80 + "\n")
    
    bert_best_params = bert_study.best_params
    timestamp_bert = datetime.now().strftime("%Y%m%d_%H%M%S")
    bert_output_dir = f"./results/bert-base-chinese_{timestamp_bert}"
    bert_model_path = f"{bert_output_dir}/final_model"
    
    ensure_dir_exists(bert_output_dir)
    ensure_dir_exists(bert_model_path)
    
    bert_training_args = TrainingArguments(
        output_dir=bert_output_dir,
        num_train_epochs=FULL_TRAIN_EPOCHS,
        per_device_train_batch_size=bert_best_params.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=4,
        learning_rate=bert_best_params.get("learning_rate", 3e-5),
        warmup_steps=bert_best_params.get("warmup_steps", 100),
        weight_decay=bert_best_params.get("weight_decay", 0.01),
        logging_dir=f"{bert_output_dir}/logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        report_to="none",
        seed=42,
        data_seed=42
    )
    
    bert_data_processor = KTDataProcessor(csv_path=DATASET_PATH)
    bert_data_processor.prepare_data(test_size=0.2, random_state=42)
    
    bert_finetuner = KTFinetuner(
        model_name=bert_model_name,
        data_processor=bert_data_processor,
        training_args=bert_training_args,
        max_token_len=512
    )
    
    bert_finetuner.run_finetuning()
    bert_finetuner.save_model(save_path=bert_model_path)
    
    # 獲取 BERT 最終評估結果
    bert_eval_results = bert_finetuner.trainer.evaluate()
    training_results["bert"] = {
        **bert_eval_results,
        "model_path": bert_model_path,
        "timestamp": timestamp_bert
    }
    
    print(f"\n✅ BERT 訓練完成！模型保存至: {bert_model_path}\n")
    
    # ========================================
    # 實驗 2: RoBERTa 模型
    # ========================================
    
    print("\n" + "="*80)
    print("🔍 實驗 2/2: RoBERTa 模型 (hfl/chinese-roberta-wwm-ext)")
    print("="*80 + "\n")
    
    roberta_model_name = "hfl/chinese-roberta-wwm-ext"
    roberta_output_base = "./optuna_results/roberta"
    
    # RoBERTa Optuna 搜索
    print(f"開始 RoBERTa 超參數搜索 ({N_TRIALS} trials)...\n")
    
    roberta_searcher = HyperparameterSearcher(
        model_class=KTFinetuner,
        data_processor_class=KTDataProcessor
    )
    
    roberta_study = roberta_searcher.run_search(
        csv_path=DATASET_PATH,
        model_name=roberta_model_name,
        output_base_dir=roberta_output_base,
        n_trials=N_TRIALS,
        study_name="roberta_hp_search"
    )
    
    # RoBERTa 完整訓練
    print("\n" + "="*80)
    print(f"🚀 使用最佳參數訓練 RoBERTa ({FULL_TRAIN_EPOCHS} epochs)")
    print("="*80 + "\n")
    
    roberta_best_params = roberta_study.best_params
    timestamp_roberta = datetime.now().strftime("%Y%m%d_%H%M%S")
    roberta_output_dir = f"./results/roberta-chinese_{timestamp_roberta}"
    roberta_model_path = f"{roberta_output_dir}/final_model"
    
    ensure_dir_exists(roberta_output_dir)
    ensure_dir_exists(roberta_model_path)
    
    roberta_training_args = TrainingArguments(
        output_dir=roberta_output_dir,
        num_train_epochs=FULL_TRAIN_EPOCHS,
        per_device_train_batch_size=roberta_best_params.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=4,
        learning_rate=roberta_best_params.get("learning_rate", 3e-5),
        warmup_steps=roberta_best_params.get("warmup_steps", 100),
        weight_decay=roberta_best_params.get("weight_decay", 0.01),
        logging_dir=f"{roberta_output_dir}/logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        report_to="none",
        seed=42,
        data_seed=42
    )
    
    roberta_data_processor = KTDataProcessor(csv_path=DATASET_PATH)
    roberta_data_processor.prepare_data(test_size=0.2, random_state=42)
    
    roberta_finetuner = KTFinetuner(
        model_name=roberta_model_name,
        data_processor=roberta_data_processor,
        training_args=roberta_training_args,
        max_token_len=512
    )
    
    roberta_finetuner.run_finetuning()
    roberta_finetuner.save_model(save_path=roberta_model_path)
    
    # 獲取 RoBERTa 最終評估結果
    roberta_eval_results = roberta_finetuner.trainer.evaluate()
    training_results["roberta"] = {
        **roberta_eval_results,
        "model_path": roberta_model_path,
        "timestamp": timestamp_roberta
    }
    
    print(f"\n✅ RoBERTa 訓練完成！模型保存至: {roberta_model_path}\n")
    
    # ========================================
    # 📊 生成比較報告
    # ========================================
    
    comparison_output = "./results/model_comparison_report.csv"
    ensure_dir_exists("./results")
    
    generate_comparison_report(
        bert_results=training_results["bert"],
        roberta_results=training_results["roberta"],
        output_path=comparison_output
    )
    
    print("\n" + "="*80)
    print("🎉 雙模型比較實驗完成！")
    print("="*80)
    print(f"📁 BERT 模型: {bert_model_path}")
    print(f"📁 RoBERTa 模型: {roberta_model_path}")
    print(f"📊 比較報告: {comparison_output}")
    print("="*80 + "\n")
```

**位置**: 第 962-1145 行  
**改動行數**: 完全重寫主程序（約 200 行）

---

## Verification Plan

### 測試階段 1: 代碼修改驗證

**目的**: 確保代碼改動後沒有語法錯誤

```bash
# 檢查語法
python -m py_compile finetune_bert.py
```

**預期結果**: 無錯誤

---

### 測試階段 2: 快速功能測試（可選）

如果想在完整訓練前快速驗證：

```python
# 修改主程序中的參數
N_TRIALS = 2  # 從 15 改為 2
FULL_TRAIN_EPOCHS = 3  # 從 60 改為 3

# 運行快速測試
python finetune_bert.py
```

**預期結果**:
- BERT 和 RoBERTa 都能正常訓練
- 生成比較報告 CSV
- 兩個模型都保存成功

---

### 測試階段 3: 完整實驗（正式運行）

```bash
# 確保參數已恢復
N_TRIALS = 15
FULL_TRAIN_EPOCHS = 60

# 背景執行（SSH 斷線也會繼續）
nohup python finetune_bert.py > training_comparison.log 2>&1 &

# 監控進度
tail -f training_comparison.log
```

**預期結果**:
- BERT Optuna 搜索 15 trials ✅
- BERT 60 epochs 訓練 ✅
- RoBERTa Optuna 搜索 15 trials ✅
- RoBERTa 60 epochs 訓練 ✅
- 生成比較報告 CSV ✅
- 兩個模型保存 ✅

**預計時間**: 2-4 小時（取決於硬體）

---

### 驗證檢查清單

完成後檢查以下文件是否存在：

```
results/
├── bert-base-chinese_YYYYMMDD_HHMMSS/
│   ├── final_model/
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── tokenizer files...
│   ├── training_metrics_visualization.png
│   ├── per_class_metrics_visualization.png
│   ├── confusion_matrix_heatmap.png
│   └── training_metrics_summary.csv
│
├── roberta-chinese_YYYYMMDD_HHMMSS/
│   ├── final_model/
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── tokenizer files...
│   ├── training_metrics_visualization.png
│   ├── per_class_metrics_visualization.png
│   ├── confusion_matrix_heatmap.png
│   └── training_metrics_summary.csv
│
└── model_comparison_report.csv  ← 比較報告

optuna_results/
├── bert/
│   ├── study_results.csv
│   ├── optimization_history.png
│   └── param_importances.png
│
└── roberta/
    ├── study_results.csv
    ├── optimization_history.png
    └── param_importances.png
```

---

## 後續步驟

完成此次實驗後，您可以：

1. **分析比較報告**：
   ```bash
   # 查看 CSV
   cat results/model_comparison_report.csv
   
   # 或用 Python 打開
   import pandas as pd
   df = pd.read_csv('results/model_comparison_report.csv')
   print(df)
   ```

2. **使用最佳模型進行 Transfer Learning**：
   ```python
   from transformers import AutoModelForSequenceClassification, AutoTokenizer
   
   # 載入最佳模型（假設是 RoBERTa）
   model_path = "results/roberta-chinese_20260110_XXXXXX/final_model"
   model = AutoModelForSequenceClassification.from_pretrained(model_path)
   tokenizer = AutoTokenizer.from_pretrained(model_path)
   ```

3. **未來擴展**（如需要）：
   - 添加 Longformer 處理長文本
   - 添加 ELECTRA、ALBERT 等其他模型
   - 只需在 `MODEL_CONFIGS` 中添加配置即可

---

## 代碼改動總結

| 修改項 | 位置 | 改動類型 | 行數 |
|--------|------|----------|------|
| 1. Import RoBERTa | 13-18 行 | 新增 | +3 |
| 2. MODEL_CONFIGS | 22 行後 | 新增 | +20 |
| 3. 重命名並通用化類別 | 607-643 行 | 修改 | ~45 |
| 4. 更新引用 | 866-871 行 | 修改 | 1 |
| 5. 比較報告函數 | 953 行後 | 新增 | +85 |
| 6. 主程序重寫 | 962-1145 行 | 重寫 | ~200 |
| **總計** | - | - | **~354 行** |

**核心理念**: 保持現有架構，添加新功能，最小化風險。

---

## 預期實驗結果

根據文獻和經驗，RoBERTa 通常在中文任務上表現優於 BERT：

- **BERT macro-F1**: 預期 ~97% (根據 README 顯示的歷史最佳 97.62%)
- **RoBERTa macro-F1**: 預期 ~98-99% (可能有 1-2% 的提升)

**關鍵優勢**:
1. RoBERTa 使用更大的訓練數據和更好的訓練策略
2. Whole Word Masking (WWM) 更適合中文
3. 動態 masking 策略

但最終結果仍需實驗驗證！🔬
