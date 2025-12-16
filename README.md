# BERT 知識追蹤 (Knowledge Tracing) 微調專案

這是一個使用 BERT 進行知識追蹤的微調專案，目標是根據學生的學習資料預測其掌握度（待加強、尚可、良好、精熟）。

## 📁 專案結構

```
finetune_KT/
│
├── 📄 finetune_bert.py              # 主程式：資料處理、Dataset、訓練邏輯
│   ├── KTDataProcessor             # 資料處理類別
│   ├── KTDynamicDataset            # PyTorch Dataset 類別
│   └── BertKTFinetuner             # 訓練器類別
│
├── 📄 view_training_history.py      # 訓練結果視覺化工具
├── 📄 requirements.txt              # Python 套件依賴
├── 📄 pytest.ini                    # pytest 配置文件
├── 📄 .gitignore                    # Git 忽略文件配置
├── 📄 README.md                     # 專案說明文件（本文件）
│
├── 📁 test/                         # 測試資料夾
│   ├── 📄 __init__.py              # 使 test 成為 Python package
│   ├── 📄 test_kt_data_processor.py    # KTDataProcessor 單元測試（10個測試案例）
│   ├── 📄 test_kt_dynamic_dataset.py   # KTDynamicDataset 單元測試（22個測試案例）
│   └── 📄 test_bert_kt_finetunner.py   # BertKTFinetuner 單元測試（5個測試案例）
│
├── 📁 datasets/                     # 資料集資料夾
│   └── 📄 finetune_dataset.csv     # 訓練資料集（需自行準備）
│
├── 📁 venv/                         # Python 虛擬環境（git ignored）
├── 📁 bert_kt_finetune_results/    # 訓練過程檔案（自動生成，git ignored）
├── 📁 my_finetuned_bert_kt_model/  # 最終訓練完成的模型（自動生成，git ignored）
└── 📁 logs/                         # TensorBoard 訓練日誌（自動生成，git ignored）
```

## 🧠 模型說明

### 使用的模型：bert-base-chinese

**基本資訊**:
- 📌 **模型名稱**: `bert-base-chinese`
- 📌 **詞彙表大小**: 21,128 個 tokens
- 📌 **最大序列長度**: 512 tokens
- 📌 **隱藏層大小**: 768
- 📌 **總參數量**: ~102M

**特點**:
- ✅ 專門針對簡體中文訓練
- ✅ 使用字符級分詞 (Character-level tokenization)
- ✅ 模型較小，訓練速度快
- ✅ 適合中文文本分類任務

### 🤔 為什麼選擇 bert-base-chinese？

1. **中文優化**: 專為中文設計，效果優於多語言模型
2. **模型大小**: 適中的模型大小，訓練和推理速度快
3. **易於使用**: Hugging Face 官方支援，文檔完整
4. **資源需求**: 對硬體要求較低，普通 GPU 即可訓練

### 🚀 其他可選模型

| 模型名稱 | 特點 | 適用場景 |
|---------|------|---------|
| `bert-base-multilingual-cased` | 支援104種語言 | 多語言混合文本 |
| `hfl/chinese-bert-wwm-ext` | 全詞遮罩，效果更好 | 追求更高準確率 |
| `hfl/chinese-roberta-wwm-ext` | RoBERTa 架構 | 長文本理解 |

## 🛠 環境設置

### Linux/Mac

```bash
# 創建虛擬環境
python3 -m venv venv

# 啟動虛擬環境
source venv/bin/activate

# 升級 pip
pip install --upgrade pip setuptools wheel

# 安裝依賴
pip install -r requirements.txt
```

## 🚀 快速開始

### 步驟 1: 準備資料集

將您的資料集放在 `datasets/finetune_dataset_k4_global.csv`，確保包含以下欄位：

| 欄位名稱 | 說明 | 範例 |
|---------|------|------|
| `chapter` | 章節名稱 | 監督式學習 |
| `section` | 知識點 | 線性迴歸 |
| `all_logs` | 作答紀錄 | ［題目1］：（簡答題）什麼是前向傳播？<br>［學生答案］：...<br>［學生表現］：部分正確 |
| `Preview_ChatLog` | 課前對話 | 老師這個概念是什麼 |
| `Review_ChatLog` | 課後對話 | 我理解了謝謝 |
| `Mastery_Level_K4` | 掌握度標籤（4等級） | 精熟 / 良好 / 尚可 / 待加強 |

### 步驟 2: 執行訓練

```bash
python finetune_bert.py
```

**訓練輸出範例**:
```
原始資料 'finetune_dataset_k4_global.csv' 讀取成功，共 995 筆
資料清理完成，剩餘 995 筆有效資料

原始資料標籤分佈：
  待加強 (label=0): 248 筆 (24.92%)
  尚可 (label=1): 249 筆 (25.03%)
  良好 (label=2): 249 筆 (25.03%)
  精熟 (label=3): 249 筆 (25.03%)

資料分割完成：訓練集 796 筆, 驗證集 199 筆

模型將在 cuda 上運行

--- 開始 Finetune ---
Epoch 1/3: {'loss': 1.63, 'accuracy': 0.90}
Epoch 2/3: {'loss': 0.21, 'accuracy': 0.94}
Epoch 3/3: {'loss': 0.11, 'accuracy': 0.97}

--- 訓練完成 ---
模型與 Tokenizer 已儲存至: ./results/bert-base-chinese_20251217_000816/final_model
```

### 步驟 3: 使用訓練好的模型

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 載入模型（使用最新的訓練結果）
model_path = "./results/bert-base-chinese_20251217_000816/final_model"  # 更換為你的模型路徑
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# 準備輸入
text = """
章節 : 監督式學習
知識點 : 線性迴歸
作答紀錄 :
答對4題答錯1題
[課前相關對話紀錄]
老師線性迴歸是什麼
[課後相關對話紀錄]
我懂了謝謝
學生掌握度 : 精熟 [MASK]
"""

# Tokenize 和預測
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()

# 結果
label_map = {0: "待加強", 1: "尚可", 2: "良好", 3: "精熟"}
print(f"預測結果: {label_map[prediction]}")
```

## 📊 訓練結果視覺化

### 訓練曲線圖解讀

訓練完成後，會在 `results/bert-base-chinese_<timestamp>/` 目錄下生成視覺化圖表。

#### training_metrics_visualization.png

此圖表包含兩個子圖，幫助您了解模型的訓練狀況：

**1. Validation Loss（驗證損失）**
- **Y軸**：Loss 值（越低越好）
- **X軸**：Epoch（訓練輪數）
- **理想曲線**：穩定下降
- **警訊**：
  - 📈 上升 → 可能過擬合
  - 📊 震盪 → 學習率可能過高
  - ➖ 平坦 → 模型可能已收斂

**2. Validation Accuracy（驗證準確率）**
- **Y軸**：準確率（0-1，越高越好）
- **X軸**：Epoch（訓練輪數）
- **理想曲線**：穩定上升
- **目標**：
  - ✅ > 0.7 (70%) - 良好
  - ✅ > 0.8 (80%) - 優秀
  - ✅ > 0.9 (90%) - 卓越

#### 範例解讀

**良好的訓練曲線特徵**：
- ✅ Loss 穩定下降（例如：0.27 → 0.11）
- ✅ Accuracy 持續上升（例如：90% → 97%）
- ✅ 曲線平滑，沒有劇烈震盪
- ✅ 沒有過擬合跡象

**需要注意的情況**：
- ⚠️ Loss 上升：可能過擬合，考慮減少 epochs
- ⚠️ Accuracy 停滯：可能需要調整學習率
- ⚠️ 曲線震盪：學習率可能過高

---

## 🔧 訓練過程詳解

### Loss Function（損失函數）

本專案使用 **Cross-Entropy Loss**（交叉熵損失）進行 4 分類任務。

#### 什麼是 Cross-Entropy Loss？

Cross-Entropy Loss 衡量模型預測的機率分佈與真實標籤之間的差異。

**數學概念**：
```
Loss = -Σ y_true * log(y_pred)
```

**4 分類範例**：
- 真實標籤：精熟 (label=3)
- 模型預測機率：[0.1, 0.2, 0.1, 0.6]
  - 待加強：10%
  - 尚可：20%
  - 良好：10%
  - 精熟：60% ✓
- Loss = -log(0.6) ≈ 0.51

**為什麼使用 Cross-Entropy？**
- ✅ 適合分類任務
- ✅ 懲罰錯誤的預測
- ✅ 鼓勵模型輸出高信心的正確預測
- ✅ 可微分，適合梯度下降優化

### 優化器（Optimizer）

使用 **AdamW** 優化器：
- **Adam**：自適應學習率優化器（Adaptive Moment Estimation）
- **W**：Weight Decay（權重衰減）正則化

**參數設定**：
```python
learning_rate: 5e-5      # 學習率（BERT fine-tuning 的標準值）
weight_decay: 0.01       # 權重衰減（防止過擬合）
```

**AdamW 的優勢**：
- ✅ 自動調整每個參數的學習率
- ✅ 對學習率不敏感
- ✅ 收斂速度快
- ✅ 適合 Transformer 模型

### 學習率調度（Learning Rate Scheduler）

使用 **Linear Warmup + Decay** 策略：

**Warmup 階段**（前 50 steps）：
- 學習率從 0 線性增加到 5e-5
- 目的：穩定訓練初期，避免梯度爆炸

**Decay 階段**（之後）：
- 學習率線性衰減到接近 0
- 目的：精細調整模型參數

**為什麼需要 Warmup？**
- ✅ 避免初期梯度過大導致不穩定
- ✅ 讓模型逐步適應資料
- ✅ 提高訓練穩定性

**學習率變化示意**：
```
5e-5 |     ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾╲
     |    ╱                  ╲
     |   ╱                    ╲
     |  ╱                      ╲___
0    |_╱__________________________|___
     0   50  100  150  200  250  300
         Warmup    Training Steps
```

### 訓練配置詳解

```python
{
    "num_train_epochs": 3,                    # 訓練 3 輪
    "per_device_train_batch_size": 4,         # 每個 GPU 的 batch size
    "per_device_eval_batch_size": 4,          # 評估時的 batch size
    "learning_rate": 5e-5,                    # 初始學習率
    "warmup_steps": 50,                       # Warmup 步數
    "weight_decay": 0.01,                     # 權重衰減
    "logging_steps": 10,                      # 每 10 步記錄一次
    "eval_strategy": "epoch",                 # 每個 epoch 結束後評估
    "save_strategy": "epoch",                 # 每個 epoch 結束後儲存
    "load_best_model_at_end": True,           # 載入最佳模型
    "metric_for_best_model": "accuracy",      # 以準確率為最佳模型指標
}
```

**參數說明**：

| 參數 | 說明 | 為什麼這樣設定 |
|------|------|---------------|
| `num_train_epochs: 3` | 訓練 3 輪 | 足夠讓模型收斂，避免過擬合 |
| `batch_size: 4` | 每次處理 4 筆資料 | 平衡記憶體使用和訓練穩定性 |
| `learning_rate: 5e-5` | 學習率 0.00005 | BERT fine-tuning 的標準值 |
| `warmup_steps: 50` | 前 50 步預熱 | 穩定訓練初期 |
| `weight_decay: 0.01` | 權重衰減 1% | 防止過擬合 |
| `eval_strategy: "epoch"` | 每輪評估 | 及時發現問題 |
| `load_best_model_at_end` | 載入最佳模型 | 確保使用最優性能的模型 |

### 評估指標

**Accuracy（準確率）**：
```
Accuracy = 正確預測數 / 總預測數
```

**範例計算**：
- 驗證集：199 筆
- 正確預測：193 筆
- Accuracy = 193/199 ≈ 0.9698 (96.98%)

**為什麼使用 Accuracy？**
- ✅ 直觀易懂
- ✅ 適合平衡資料集
- ✅ 與實際應用場景一致

### 防止過擬合的機制

1. **Weight Decay (0.01)**
   - 懲罰過大的權重
   - 鼓勵模型簡化
   - L2 正則化的變體

2. **Early Stopping**
   - 載入驗證集表現最好的模型
   - 避免訓練過久
   - 自動選擇最佳 checkpoint

3. **Stratified Split**
   - 訓練集和驗證集保持相同的標籤分佈
   - 確保評估公平
   - 避免資料偏差

4. **Dropout（BERT 內建）**
   - 訓練時隨機丟棄部分神經元
   - 增強模型泛化能力
   - 防止過度依賴特定特徵

### Epoch 小數點說明

訓練過程中，您可能會看到 `epoch: 0.1`, `epoch: 0.2` 等小數點：

**為什麼會有小數點？**
- 每隔 10 個 steps 記錄一次訓練指標
- 小數點表示當前 epoch 的完成進度
- 例如：`epoch: 1.5` 表示第 2 個 epoch 完成了 50%

**計算方式**：
```
當前 epoch = 已處理樣本數 / 總訓練樣本數
```

**範例**：
- 總訓練樣本：796 筆
- Batch size：4
- 每個 epoch 的 steps：796 ÷ 4 = 199 steps
- Step 100/199 → epoch ≈ 1.0（第 1 個 epoch 完成）
- Step 110/199 → epoch ≈ 1.1（第 2 個 epoch 的 10%）

---

## ❓ 常見問題

### Q1: 為什麼訓練準確率是 100% 但驗證準確率只有 97%？

**A**: 這是正常現象，稱為「過擬合」的輕微跡象。
- **訓練集**：模型已經看過，容易記住
- **驗證集**：模型沒看過，測試真實能力
- **97% 的驗證準確率已經非常優秀**

**如何判斷是否過擬合？**
- ✅ 正常：訓練和驗證準確率都很高（如 100% vs 97%）
- ⚠️ 過擬合：訓練準確率很高，驗證準確率很低（如 100% vs 60%）

### Q2: Loss 和 Accuracy 哪個更重要？

**A**: 兩者都重要，但關注點不同：
- **Loss**：優化目標，越低越好，反映模型的學習進度
- **Accuracy**：實際性能，更直觀，反映模型的預測能力
- **建議**：Loss 下降 + Accuracy 上升 = 訓練成功 ✅

### Q3: 如何判斷模型是否過擬合？

**A**: 觀察訓練曲線：
- ✅ **正常**：訓練和驗證 Loss 都下降
- ⚠️ **過擬合**：訓練 Loss 下降，驗證 Loss 上升
- **解決方法**：
  - 增加 `weight_decay`（例如：0.01 → 0.05）
  - 減少 `epochs`（例如：5 → 3）
  - 增加訓練資料

### Q4: 為什麼有些資料超過 512 tokens？

**A**: BERT 的最大輸入長度是 512 tokens。
- **超過的部分會被截斷**
- **26% 的資料超過 512 是可接受的**
- **替代方案**：
  - 使用 Longformer（支援 4096 tokens）
  - 使用 BigBird（支援更長文本）
  - 調整資料預處理策略

### Q5: 可以調整哪些參數來改善性能？

**A**: 常見調整策略：

**增加訓練輪數**：
```python
"num_train_epochs": 5,  # 從 3 增加到 5
```

**調整學習率**：
```python
"learning_rate": 3e-5,  # 從 5e-5 降低（更穩定）
```

**增加 Batch Size**（需要更多 GPU 記憶體）：
```python
"per_device_train_batch_size": 8,  # 從 4 增加到 8
```

**增加資料量**：
- 收集更多訓練資料
- 使用資料增強技術

### Q6: 訓練時出現 "CUDA out of memory" 怎麼辦？

**A**: 降低記憶體使用：
```python
"per_device_train_batch_size": 2,  # 從 4 降到 2
"per_device_eval_batch_size": 2,   # 從 4 降到 2
```

或使用梯度累積：
```python
"gradient_accumulation_steps": 2,  # 累積 2 步再更新
```

### Q7: 如何解讀訓練日誌中的 epoch 小數點？

**A**: 小數點表示訓練進度：
- `epoch: 0.1` = 第 1 個 epoch 的 10%
- `epoch: 1.0` = 第 1 個 epoch 完成
- `epoch: 2.5` = 第 3 個 epoch 的 50%

這是正常且有用的設計，幫助您即時監控訓練進度。


## 🧪 測試

### 執行測試

```bash
# 執行所有測試
pytest test/ -v

# 執行特定測試文件
pytest test/test_kt_data_processor.py -v
pytest test/test_kt_dynamic_dataset.py -v
pytest test/test_bert_kt_finetunner.py -v

# 顯示測試覆蓋率
pytest test/ --cov=finetune_bert --cov-report=html

# 顯示 print 輸出（用於除錯）
pytest test/ -v -s
```

### 📊 測試覆蓋範圍（37個測試案例）

#### test/test_kt_data_processor.py (10個測試)
- ✅ 初始化屬性驗證
- ✅ 成功載入和清理資料
- ✅ 無效標籤過濾
- ✅ NaN 值處理
- ✅ 標籤映射準確性
- ✅ 資料分割邏輯
- ✅ 缺少欄位處理
- ✅ 流程控制
- ✅ 檔案不存在異常
- ✅ 分層抽樣分佈驗證

#### test/test_kt_dynamic_dataset.py (22個測試)
- ✅ 初始化和基本屬性
- ✅ 資料結構和 tensor 形狀
- ✅ 文本格式化和組合
- ✅ Tokenization 正確性
- ✅ 特殊 token 處理（[CLS], [SEP]）
- ✅ Padding 和 attention mask
- ✅ 文本截斷
- ✅ 空欄位處理
- ✅ 不同 max_length 設定
- ✅ 索引處理和邊界測試
- ✅ 中文文本編碼

#### test/test_bert_kt_finetunner.py (5個測試)
- ✅ 初始化流程
- ✅ 計算 metrics
- ✅ 訓練流程（含 accuracy 曲線）
- ✅ 儲存模型流程
- ✅ 儲存模型例外處理

### 測試輸出範例

```
$ pytest test/ -v
======================== test session starts ========================
collected 37 items

test/test_kt_data_processor.py::TestKTDataProcessor::test_init_attributes PASSED
test/test_kt_data_processor.py::TestKTDataProcessor::test_load_and_clean_success PASSED
...
test/test_kt_dynamic_dataset.py::TestKTDynamicDataset::test_text_formatting PASSED
test/test_kt_dynamic_dataset.py::TestKTDynamicDataset::test_chinese_text_encoding PASSED
...
test/test_bert_kt_finetunner.py::test_run_finetuning PASSED

======================== 37 passed in 25.43s ========================
```

## 🔧 常見問題

### 1. ImportError: accelerate>=0.26.0 required

**問題**: `transformers>=4.37.0` 需要 `accelerate>=0.26.0`

**解決方案**:
```bash
pip install 'accelerate>=0.26.0' --upgrade
```

### 2. CUDA out of memory

**解決方案**:
```python
# 在 finetune_bert.py 中調整批次大小
training_args_dict = {
    "per_device_train_batch_size": 2,  # 從 4 降到 2
    "per_device_eval_batch_size": 4,   # 從 8 降到 4
}
```

### 3. 訓練速度慢

**解決方案**:
- ✅ 使用 GPU（CUDA）
- ✅ 減少 `max_token_len`（預設 512，可降至 256）
- ✅ 使用混合精度訓練：
  ```python
  training_args_dict = {
      "fp16": True,  # 需要支援的 GPU
  }
  ```

### 4. TypeError: TrainingArguments got unexpected keyword 'evaluation_strategy'

**問題**: transformers ≥4.30 改用 `eval_strategy` 參數

**解決方案**:
```python
# ✅ 正確（新版）
training_args_dict = {
    "eval_strategy": "epoch",
}

# ❌ 錯誤（舊版）
training_args_dict = {
    "evaluation_strategy": "epoch",  # 已棄用
}
```

## 📝 訓練參數說明

| 參數名稱 | 預設值 | 說明 | 建議範圍 |
|---------|--------|------|---------|
| `num_train_epochs` | 3 | 訓練輪數 | 3-10 |
| `per_device_train_batch_size` | 4 | 訓練批次大小 | 2-32 |
| `per_device_eval_batch_size` | 8 | 評估批次大小 | 4-64 |
| `warmup_steps` | 50 | 預熱步數 | 0-500 |
| `weight_decay` | 0.01 | 權重衰減 | 0-0.1 |
| `eval_strategy` | "epoch" | 評估策略 | epoch/steps |
| `save_strategy` | "epoch" | 儲存策略 | epoch/steps |
| `load_best_model_at_end` | True | 載入最佳模型 | True/False |

## 💻 系統需求

### 硬體需求

| 組件 | 最低需求 | 建議配置 |
|-----|---------|---------|
| **CPU** | 2 核心 | 4 核心以上 |
| **RAM** | 8GB | 16GB+ |
| **GPU** | 無（可用 CPU） | NVIDIA GPU 6GB+ VRAM |
| **儲存空間** | 5GB | 10GB+ |

### 軟體需求

- **Python**: 3.8+ （建議 3.10+）
- **CUDA**: 可選，11.8+ （GPU 加速）
- **作業系統**: Linux / macOS / Windows 10+

## 📄 授權

本專案僅供學術研究使用。

---

**最後更新**: 2024-01  
**專案版本**: v2.0.0  
**Python 版本**: 3.8+  
**PyTorch 版本**: 2.2.0+  
**Transformers 版本**: 4.37.0+  
**測試覆蓋**: 37個測試案例
