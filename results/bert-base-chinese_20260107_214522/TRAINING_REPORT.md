# BERT Fine-tuning 訓練完成報告

**日期時間**: 2026-01-07 21:45:22  
**訓練資料集**: finetune_dataset_1132_v2.csv (725 筆，3 個掌握度等級)  
**訓練時長**: 342 秒 (~5.7 分鐘)

---

## 📊 最終成果

### 整體表現
- **最佳驗證準確率**: **96.55%** (Epoch 7)
- **最終驗證準確率**: 96.55% (Epoch 10，使用最佳模型)
- **最終驗證 Loss**: 0.1226

### 每個類別表現（最佳模型 - Epoch 7）

| 類別 | Precision | Recall | F1-Score | Accuracy |
|------|-----------|--------|----------|----------|
| **待加強** | 100.0% | 94.9% | 97.4% | 94.9% |
| **尚可** | 100.0% | 97.1% | 98.5% | 97.1% |
| **精熟** | 78.3% | 100.0% | 87.8% | 100.0% |

### 訓練效率
- **訓練速度**: 16.96 samples/sec
- **總訓練步數**: 730 steps (10 epochs × 73 steps)
- **檢查點數量**: 10 個（每個 epoch 保存一次）

---

## 📈 訓練進程分析

### Epoch-by-Epoch 進展

| Epoch | Val Accuracy | Val Loss | 待加強 F1 | 尚可 F1 | 精熟 F1 |
|-------|--------------|----------|-----------|---------|---------|
| 1 | 88.97% | 0.314 | 93.7% | 92.9% | 61.5% |
| 2 | 85.52% | 0.535 | 91.3% | 91.0% | 0.0% ⚠️ |
| 3 | 91.72% | 0.237 | 95.6% | 94.2% | 71.8% |
| 4 | 89.66% | 0.321 | 95.6% | 92.2% | 69.4% |
| 5 | 94.48% | 0.170 | 95.6% | 97.7% | 81.8% |
| 6 | 95.17% | 0.181 | 97.4% | 96.4% | 83.3% |
| **7** | **96.55%** | **0.123** | **97.4%** | **98.5%** | **87.8%** ⭐ |
| 8 | 91.03% | 0.430 | 92.8% | 95.7% | 59.3% |
| 9 | 91.72% | 0.372 | 93.5% | 95.7% | 64.3% |
| 10 | 94.48% | 0.180 | 97.5% | 95.6% | 80.0% |

> [!NOTE]
> **最佳模型選擇**
> 
> 系統自動選擇 Epoch 7 作為最佳模型（準確率 96.55%），並保存於 `final_model/` 目錄。

---

## 🎯 關鍵發現

### 1. 類別不平衡影響

**資料分佈**:
- 待加強: 296 筆 (40.83%)
- 尚可: 339 筆 (46.76%)
- 精熟: 90 筆 (12.41%) ⚠️

**影響**:
- 「精熟」類別樣本最少，導致其 F1-score 相對較低 (87.8%)
- 但 Recall 達到 100%，表示模型不會漏掉精熟學生
- Precision 較低 (78.3%)，可能會將部分「尚可」誤判為「精熟」

### 2. 訓練穩定性

- Epoch 2 出現異常：精熟類別 F1 = 0%（模型暫時無法識別該類別）
- Epoch 3 開始恢復，之後逐步提升
- Epoch 7-10 之間表現穩定，但略有波動

### 3. 長文本處理

**提醒**: 
- 24.69% 的資料超過 512 tokens（最大 11,746 tokens）
- 目前使用截斷策略處理超長文本
- 可能損失部分對話記錄資訊

---

## 📁 輸出檔案

### 訓練結果目錄
`results/bert-base-chinese_20260107_214522/`

### 主要檔案

#### 1. 最終模型
- 📂 `final_model/` - 最佳模型檢查點（Epoch 7）
  - `config.json` - 模型配置
  - `pytorch_model.bin` - 模型權重
  - `tokenizer_config.json` - Tokenizer 配置
  - `vocab.txt` - 詞彙表

#### 2. 視覺化圖表
- 📊 `training_metrics_visualization.png` - 訓練曲線（Loss、Accuracy）
- 📊 `per_class_metrics_visualization.png` - 每個類別的指標曲線

#### 3. 數據摘要
- 📄 `training_metrics_summary.csv` - 完整訓練指標記錄

#### 4. 檢查點
- 📂 `checkpoint-{73,146,219,...,730}/` - 10 個訓練檢查點

---

## 💡 後續建議

### 1. 模型優化

#### A. 處理類別不平衡
```python
# 在 TrainingArguments 中加入類別權重
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.array([0, 1, 2]),
    y=train_labels
)
```

#### B. 資料增強
- 對「精熟」類別進行資料增強（回翻譯、同義詞替換等）
- 目標：增加精熟類別樣本至至少 150-200 筆

### 2. 長文本模型評估

由於 24.69% 資料超過 512 tokens，建議測試：

| 模型 | Max Length | 優勢 | 預期改善 |
|------|------------|------|----------|
| **Longformer** | 4096 | Sparse attention，記憶體效率高 | +2-3% accuracy |
| **BigBird** | 4096 | Random+Local+Global attention | +2-4% accuracy |
| **RoBERTa-wwm-ext-large** | 512 | 更強的中文理解能力 | +1-2% accuracy |

### 3. 超參數調優

當前配置 vs 建議調整：

| 參數 | 當前值 | 建議值 | 原因 |
|------|--------|--------|------|
| `num_train_epochs` | 3 | 5-7 | Epoch 7 表現最佳 |
| `learning_rate` | 5e-5 | 3e-5 或 2e-5 | 減少訓練波動 |
| `warmup_steps` | 50 | 100 | 更平滑的學習率調整 |
| `per_device_train_batch_size` | 4 | 8 或 16 | 加快訓練速度 |

### 4. 評估優化

建議增加評估指標：
- **加權 F1-score**: 考慮類別不平衡
- **混淆矩陣**: 分析誤判模式
- **Per-sample 分析**: 找出困難樣本特徵

---

## 🚀 使用訓練好的模型

### 載入模型進行預測

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 載入模型
model_path = "results/bert-base-chinese_20260107_214522/final_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

# 準備輸入
text = """
章節 : A_機器學習-監督式學習
知識點 : 監督式學習的定義
學生掌握度 : [MASK]
簡答題作答紀錄 :
［題目］：什麼是監督式學習？
［學生答案］：使用標記數據訓練模型
［學生表現］：Correct
對話紀錄 :
[學生]: 監督式學習和非監督式學習的差異是什麼？
[AI Tutor]: 監督式學習使用有標籤的數據...
"""

# Tokenize 和預測
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)
    
# 解析結果
label_map = {0: "待加強", 1: "尚可", 2: "精熟"}
print(f"預測掌握度: {label_map[predictions.item()]}")
```

---

## 📝 總結

✅ **成功達成目標**:
- 3 個掌握度等級的分類器訓練完成
- 驗證準確率達到 96.55%
- 所有類別 F1-score 均超過 85%

⚠️ **需要注意**:
- 精熟類別樣本較少，建議增加資料
- 長文本截斷可能影響效能
- 建議評估長文本模型

🎯 **下一步**:
1. 在測試集上驗證模型表現
2. 分析誤判案例
3. 考慮部署或進一步優化
