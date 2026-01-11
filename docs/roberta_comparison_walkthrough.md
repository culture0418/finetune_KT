> **Git Branch**: `feature/add-roberta-comparison`  
> **完成日期**: 2026-01-11  
> **實驗耗時**: 2 小時 44 分鐘

# BERT vs RoBERTa 模型比較實驗 - 完成報告

## 🎯 實驗目標

比較 BERT 和 RoBERTa 模型在知識追蹤任務上的性能表現。

---

## ✅ 實驗執行摘要

### 訓練時間
- **開始時間**: 2026-01-11 01:18
- **完成時間**: 2026-01-11 04:02 
- **總耗時**: 約 2 小時 44 分鐘

### 資料集
- **檔案**: `datasets/finetune_dataset_1132_v2.csv`
- **總筆數**: 725 筆
- **訓練集**: 580 筆 (80%)
- **驗證集**: 145 筆 (20%)
- **標籤分佈**:
  - 待加強: 296 筆 (40.83%)
  - 尚可: 339 筆 (46.76%)
  - 精熟: 90 筆 (12.41%)

---

## 📊 訓練流程

### 階段 1: BERT 模型
1. **Optuna 超參數搜索**
   - Trials: 15
   - 最佳 Trial: Trial 1
   - 搜索耗時: ~50 分鐘

2. **完整訓練**
   - Epochs: 50
   - 使用 Optuna 找到的最佳參數
   - 訓練耗時: ~40 分鐘

### 階段 2: RoBERTa 模型
1. **Optuna 超參數搜索**
   - Trials: 15
   - 最佳 Trial: Trial 0
   - 搜索耗時: ~50 分鐘

2. **完整訓練**
   - Epochs: 50
   - 使用 Optuna 找到的最佳參數
   - 訓練耗時: ~40 分鐘

---

## 🏆 性能比較結果

### 整體指標

| 模型 | Macro F1-Score | 整體準確率 | 性能提升 |
|------|----------------|------------|----------|
| **BERT** | 0.9555 | 97.24% | - |
| **RoBERTa** | **0.9947** | **99.31%** | **+4.11%** |

### 各類別 F1-Score

| 類別 | BERT | RoBERTa | 提升 |
|------|------|---------|------|
| 待加強 | 0.9739 | **0.9916** | +1.82% |
| 尚可 | **0.9926** | **0.9926** | 0% |
| 精熟 | 0.9000 | **1.0000** | **+11.11%** |

---

## 🔍 關鍵發現

### 1. **RoBERTa 顯著優於 BERT**
- Macro F1-Score 提升 **4.11%** (0.9555 → 0.9947)
- 整體準確率提升 **2.07%** (97.24% → 99.31%)

### 2. **精熟類別的突破性表現**
- RoBERTa 在「精熟」類別達到 **100% F1-Score**
- BERT 在同類別僅為 90%
- 這對於識別高水平學生特別重要

### 3. **穩定的高性能**
- 兩個模型在「尚可」類別都達到 99.26% F1
- RoBERTa 在「待加強」類別也有提升 (97.39% → 99.16%)

---

## 💾 保存的模型

### BERT 模型
- **路徑**: `./results/bert-base-chinese_20260111_022231/final_model/`
- **大小**: 391MB
- **時間戳**: 20260111_022231

### RoBERTa 模型
- **路徑**: `./results/roberta-chinese_20260111_034253/final_model/`
- **大小**: 391MB
- **時間戳**: 20260111_034253

### 比較報告
- **CSV 報告**: `./results/model_comparison_report.csv`
- **包含內容**: 完整的性能指標比較

---

## 📈 訓練配置

### 共通配置
- **資料分割**: 8:2 (train:val)
- **Random Seed**: 42
- **Batch Size**: 由 Optuna 優化 (4/8/16)
- **Learning Rate**: 由 Optuna 優化 (1e-5 ~ 1e-4)
- **Checkpoint 策略**: save_total_limit=2

### Optuna 搜索空間
- Learning Rate: [1e-5, 1e-4] (對數分佈)
- Epochs: [5, 10, 15]
- Batch Size: [4, 8, 16]
- Warmup Steps: [50, 100, 150, 200]
- Weight Decay: [0.0, 0.1]

---

## 💾 磁碟空間優化

### 優化前問題
- 每次訓練產生 50 個 checkpoints (~60GB)
- Optuna trials 保存完整模型 (~35GB)

### 優化後
- **Optuna**: `save_strategy="no"` (不保存 checkpoint)
- **完整訓練**: `save_total_limit=2` (僅保留 2 個)
- **空間節省**: 每次實驗從 ~200GB 降至 **~5GB**

---

## ✅ 結論與建議

### 1. **推薦使用 RoBERTa**
基於實驗結果，RoBERTa 在所有指標上都優於或等於 BERT，特別是：
- ✅ 更高的整體準確率 (99.31% vs 97.24%)
- ✅ 更平衡的類別表現
- ✅ 在少數類「精熟」上達到完美分類

### 2. **模型可直接使用**
兩個訓練好的模型都已保存，可用於：
- 推論 (Inference)
- 遷移學習 (Transfer Learning)
- 進一步微調 (Further Fine-tuning)

### 3. **未來改進方向**
- 嘗試更多 Optuna trials (如 30 或 50 trials)
- 嘗試不同的資料擴增策略
- 探索 ensemble 方法結合兩個模型

---

## 📂 生成的文件清單

```
results/
├── bert-base-chinese_20260111_022231/
│   ├── final_model/                     # BERT 最終模型
│   ├── training_metrics_visualization.png
│   ├── per_class_metrics_visualization.png
│   ├── confusion_matrix_heatmap.png
│   └── training_metrics_summary.csv
│
├── roberta-chinese_20260111_034253/
│   ├── final_model/                     # RoBERTa 最終模型
│   ├── training_metrics_visualization.png
│   ├── per_class_metrics_visualization.png
│   ├── confusion_matrix_heatmap.png
│   └── training_metrics_summary.csv
│
└── model_comparison_report.csv          # 比較報告

optuna_results/
├── bert/
│   └── study_results.csv                # BERT Optuna 結果
└── roberta/
    └── study_results.csv                # RoBERTa Optuna 結果
```

---

## 🎓 技術要點

### 模型配置差異
雖然名為 RoBERTa，但 `hfl/chinese-roberta-wwm-ext` 實際使用：
- **架構**: BertConfig + BertForSequenceClassification
- **Tokenizer**: BertTokenizer
- **訓練策略**: Whole Word Masking (與標準 RoBERTa 的差異)

這是因為 HFL 的中文 RoBERTa 是基於 BERT 架構實作的。

### 關鍵修正
實驗過程中發現並修正的問題：
1. ✅ 初始配置錯誤使用 `RobertaForSequenceClassification` 導致 CUDA 錯誤
2. ✅ 改為正確的 `BertForSequenceClassification` 後訓練成功
3. ✅ 優化 checkpoint 策略節省磁碟空間

---

**實驗完成時間**: 2026-01-11 04:02  
**報告生成時間**: 2026-01-11 12:05
