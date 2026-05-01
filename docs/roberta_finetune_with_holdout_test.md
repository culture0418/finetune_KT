> **Git Branch**: `feature/llm-comparison-visualization`
> **完成日期**: 2026-04-25
> **訓練耗時**: 約 10 分鐘（Optuna 8 分鐘 + 完整訓練 ~1.5 分鐘）
> **硬體**: NVIDIA RTX 4090 (24 GB)

# RoBERTa 重新訓練（導入 held-out test set）

## 🎯 動機

舊流程（branch `feature/add-roberta-comparison`）只切 train/val 兩份，讓 LLM 評測沿用同一份 val set，導致 RoBERTa 與 LLM 比較**不公平**——RoBERTa 在訓練期間透過 early stopping「看過」val set 的訊號，但 LLM 完全沒看過。

本次重新訓練的核心改動：
1. 將原始資料額外切出 **15% test set**，對 RoBERTa 訓練完全 held-out
2. RoBERTa 與所有 LLM 統一在這份 test set 上做最終評測
3. 切分結果以 CSV 形式 commit 進 git，保證可重現

---

## 📂 資料集與切分策略

### 來源
- **檔案**: `datasets/finetune_dataset_1142_v4_without_chat_0227.csv`
- **總筆數**: 474（CSV 行數 3405 是因為 `Short_Answer_Log` 含換行）

### 切分（[split_dataset.py](../split_dataset.py)）
- **比例**: 70 / 15 / 15
- **Random seed**: 42
- **Stratify by**: `Mastery_Label`（保證三類別比例一致）
- **兩階段切分**：先切 15% test，再從剩餘 85% 切 ~17.65% 當 val
- **輸出**:
  - `datasets/splits/0227/train.csv`
  - `datasets/splits/0227/val.csv`
  - `datasets/splits/0227/test.csv`
  - `datasets/splits/0227/split_info.json`（記錄 seed、實際比例、各類別計數）

### 切分結果

| Split | 筆數 | 比例 | 待加強 | 尚可 | 精熟 |
|---|---|---|---|---|---|
| Train | 331 | 69.83% | 33 (10.0%) | 235 (71.0%) | 63 (19.0%) |
| Val | 71 | 14.98% | 7 (9.9%) | 51 (71.8%) | 13 (18.3%) |
| Test | 72 | 15.19% | 7 (9.7%) | 51 (70.8%) | 14 (19.4%) |

⚠️ 資料量偏小且**類別不平衡**（尚可 71% / 精熟 19% / 待加強 10%）— 後續訓練必須處理 class imbalance。

---

## 🧠 模型架構

| 項目 | 設定 |
|---|---|
| 預訓練模型 | `hfl/chinese-roberta-wwm-ext`（哈工大訊飛 12 層中文 RoBERTa with Whole Word Masking） |
| 下游 head | `AutoModelForSequenceClassification`（線性分類頭，`num_labels=3`） |
| Tokenizer | `BertTokenizer`，`max_length=512`，`padding="max_length"`，`truncation=True` |
| 輸入文本 | `chapter` + `section` + `Short_Answer_Log` 三欄拼接 |

---

## ⚙️ 訓練策略

### 1. 處理類別不平衡 — Weighted Cross-Entropy

自定義 `WeightedTrainer` 子類覆寫 `compute_loss`，類別權重從 train set 計算：

```
class_weight[i] = total_samples / (num_classes × class_count[i])
```

| 類別 | Train count | 權重 |
|---|---|---|
| 待加強 | 33 | **3.34** |
| 尚可 | 235 | 0.47 |
| 精熟 | 63 | 1.75 |

### 2. Optuna 超參數搜索

- **Trials**: 15
- **Pruner**: `MedianPruner(n_startup_trials=3, n_warmup_steps=5)`
- **目標**: 最大化 val macro_f1
- **耗時**: ~8 分鐘（其中 6 個 trial 被 prune、9 個完整跑完）

#### 搜索空間

| 超參數 | 範圍 / 候選 |
|---|---|
| `learning_rate` | [1e-5, 1e-4]，log-uniform |
| `num_train_epochs` | {5, 10, 15} |
| `per_device_train_batch_size` | {4, 8, 16} |
| `warmup_steps` | {50, 100, 150, 200} |
| `weight_decay` | [0.0, 0.1]，uniform |

#### 最佳 trial（#3，val macro_f1 = 0.9678）

| 超參數 | 值 |
|---|---|
| learning_rate | 3.888 × 10⁻⁵ |
| num_train_epochs | 5 |
| per_device_train_batch_size | 8 |
| warmup_steps | 50 |
| weight_decay | 0.0215 |

完整 trial 紀錄：`optuna_results/roberta/study_results.csv`

### 3. 完整訓練（基於 best params 的保守調整）

為了避免過擬合與訓練不穩定，在 Optuna 找到的最佳值上做兩處保守修正：

| 參數 | Optuna best | 完整訓練值 | 備註 |
|---|---|---|---|
| learning_rate | 3.888 × 10⁻⁵ | **× 0.5 = 1.944 × 10⁻⁵** | 降低避免 overshoot |
| warmup_steps | 50 | **+ 100 = 150** | 加長 warmup |
| num_train_epochs | 5 | **15** | 配 early stopping |

#### 其他關鍵設定

| 項目 | 值 |
|---|---|
| `eval_strategy` / `save_strategy` | `epoch` |
| `load_best_model_at_end` | `True`（依 macro_f1 選最佳 checkpoint） |
| `metric_for_best_model` | `macro_f1`（greater_is_better=True） |
| `EarlyStoppingCallback` | `patience=3` |
| `save_total_limit` | 2 |
| `seed` / `data_seed` | 42 / 42 |

---

## 📊 訓練結果

### 各 epoch val 指標

| Epoch | Eval Loss | Accuracy | 待加強 F1 | 尚可 F1 | 精熟 F1 |
|---|---|---|---|---|---|
| 1 | 1.032 | 0.704 | 0.000 | 0.826 | 0.000 |
| 2 | 0.865 | 0.761 | 0.750 | 0.851 | 0.400 |
| 3 | 0.380 | 0.915 | 0.857 | 0.952 | 0.783 |
| 4 | 0.293 | 0.958 | 0.857 | 0.980 | 0.923 |
| **5** ⭐ | **0.196** | **0.972** | **0.857** | **0.990** | **0.963** |
| 6 | 0.439 | 0.930 | 0.857 | 0.962 | 0.833 |
| 7 | 0.373 | 0.958 | 0.857 | 0.980 | 0.923 |
| 8 | 0.283 | 0.958 | 0.857 | 0.980 | 0.929 |

**早停於 epoch 8**（`patience=3`，自 epoch 5 最佳值算起）；`load_best_model_at_end=True` 將 final model 還原成 epoch 5 的 checkpoint。

### Final model 在 val set 上的表現

| 指標 | 值 |
|---|---|
| Accuracy | 0.972 |
| **Macro F1** | **0.937** |
| 待加強 P / R / F1 | 0.857 / 0.857 / 0.857 |
| 尚可 P / R / F1 | 1.000 / 0.980 / 0.990 |
| 精熟 P / R / F1 | 0.929 / 1.000 / 0.963 |

⚠️ 此處數字為 **val set**（用於早停），並非最終跟 LLM 公平比較的結果。Test set 表現由 [llm_comparison.py](../llm_comparison.py) 統一產出。

---

## 💾 產物

| 項目 | 路徑 |
|---|---|
| Final model | `results/roberta-chinese_20260425_163056/final_model/` |
| Checkpoints | `results/roberta-chinese_20260425_163056/checkpoint-{210,336}/` |
| 訓練 metrics CSV | `results/roberta-chinese_20260425_163056/training_metrics_summary.csv` |
| 訓練曲線圖 | `results/roberta-chinese_20260425_163056/training_metrics_visualization.png` |
| 各類別指標圖 | `results/roberta-chinese_20260425_163056/per_class_metrics_visualization.png` |
| 混淆矩陣 | `results/roberta-chinese_20260425_163056/confusion_matrix_heatmap.png` |
| Optuna study | `optuna_results/roberta/study_results.csv` |
| 訓練 log | `logs/finetune_roberta_20260425_162254.log` |

---

## 🔁 可重現性

```bash
# 0. 環境
source venv/bin/activate

# 1. （已完成、CSV 已 commit）切分資料集
python split_dataset.py
#    --csv  datasets/finetune_dataset_1142_v4_without_chat_0227.csv
#    --output-dir  datasets/splits/0227
#    --test-size 0.15  --val-size 0.15  --random-state 42

# 2. RoBERTa Optuna search + 完整訓練
python finetune_bert.py
#    讀取  datasets/splits/0227/{train,val}.csv
#    產出  results/roberta-chinese_<timestamp>/
```

固定 `random_state=42` + 固定 split CSV → 相同硬體上應可完整重現結果。

---

## 🆚 與舊流程的差異

| 面向 | 舊流程（roberta_comparison_walkthrough） | 本次 |
|---|---|---|
| 切分 | 80/20，in-memory 每次重切 | **70/15/15**，固定 CSV |
| Test set | 無（val set 兩用） | **獨立 held-out** |
| 與 LLM 比較公平性 | RoBERTa 看過 eval data | RoBERTa 與 LLM 同一份 unseen test |
| 切分腳本 | 無 | [split_dataset.py](../split_dataset.py) |
| Class weights | 已使用 | 沿用 |
| Optuna 設定 | 同 | 同 |
| 完整訓練 epochs | 50 | 15 + early stopping |

舊版產物保留於：
- `paper_result_old_valset_eval/`
- `results/llm_comparison_old_valset_eval/`
- `results/roberta-chinese_20260227_193714/`（舊 RoBERTa model，**不能**用於新 test set 評測，因訓練資料涵蓋部分新 test 樣本）
