# BERT 知識追蹤 (Knowledge Tracing) 微調專案

這是一個使用 BERT 進行知識追蹤的微調專案，目標是根據學生的學習資料預測其掌握度（待加強、尚可、精熟）。

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

### 系統需求

- **Python**: 3.8 或以上（建議 3.10+）
- **PyTorch**: 2.2.0 或以上
- **CUDA**: 可選，用於 GPU 加速（建議 11.8 或以上）
- **RAM**: 建議 16GB 以上
- **儲存空間**: 至少 10GB

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

### Windows

```cmd
# 創建虛擬環境
python -m venv venv

# 啟動虛擬環境
venv\Scripts\activate

# 升級 pip
python -m pip install --upgrade pip setuptools wheel

# 安裝依賴
pip install -r requirements.txt
```

## 📦 套件版本說明

本專案使用以下核心套件版本（2024年最新穩定版）：

| 套件 | 版本 | 說明 |
|------|------|------|
| `torch` | ≥2.2.0 | PyTorch 深度學習框架 |
| `transformers` | ≥4.37.0 | Hugging Face Transformers |
| `tokenizers` | ≥0.15.0 | 快速 tokenization 引擎 |
| `datasets` | ≥2.17.0 | Hugging Face 資料集工具 |
| `accelerate` | ≥0.26.0 | 分散式訓練加速（必要） |
| `pandas` | ≥2.2.0,<3.0.0 | 資料處理 |
| `numpy` | ≥1.26.0,<2.0.0 | 數值計算 |
| `scikit-learn` | ≥1.4.0 | 機器學習工具 |
| `pyarrow` | ≥15.0.0 | 高效資料讀取 |
| `matplotlib` | (可選) | 繪製訓練曲線 |

### ⚠️ 重要提示

1. **transformers ≥4.37.0 強制需要 accelerate ≥0.26.0**
   - 若未安裝 accelerate，會出現 `ImportError: accelerate>=0.26.0 required`
   - 解決方式：`pip install 'accelerate>=0.26.0'`

2. **確保使用最新版 pip**
   - 舊版 pip 可能無法正確解析套件相依性
   - 執行：`pip install --upgrade pip`

3. **GPU 支援**
   - 若要使用 GPU，請確認已安裝對應的 CUDA 版本
   - 檢查：`python -c "import torch; print(torch.cuda.is_available())"`

### 驗證安裝

```bash
# 檢查套件版本
pip list | grep -E "torch|transformers|accelerate|datasets"

# 驗證相依性（應無錯誤）
pip check

# 測試核心套件 import
python -c "from transformers import BertTokenizer; print('✓ transformers OK')"
python -c "import accelerate; print('✓ accelerate OK')"
python -c "import torch; print('✓ torch OK')"
python -c "from datasets import load_dataset; print('✓ datasets OK')"
```

預期輸出：
```
✓ transformers OK
✓ accelerate OK
✓ torch OK
✓ datasets OK
```

## 🚀 快速開始

### 步驟 1: 準備資料集

將您的資料集放在 `datasets/finetune_dataset.csv`，確保包含以下欄位：

| 欄位名稱 | 說明 | 範例 |
|---------|------|------|
| `chapter` | 章節名稱 | 監督式學習 |
| `section` | 知識點 | 線性迴歸 |
| `all_logs` | 作答紀錄 | 答對5題答錯1題 |
| `Preview_ChatLog` | 課前對話 | 老師這個概念是什麼 |
| `Review_ChatLog` | 課後對話 | 我理解了謝謝 |
| `Mastery_Level` | 掌握度標籤 | 精熟 / 尚可 / 待加強 |

### 步驟 2: 執行訓練

```bash
python finetune_bert.py
```

**訓練輸出範例**:
```
原始資料 'finetune_dataset.csv' 讀取成功，共 250 筆
資料清理完成，剩餘 240 筆有效資料

原始資料標籤分佈：
  待加強 (label=0): 80 筆 (33.33%)
  尚可 (label=1): 80 筆 (33.33%)
  精熟 (label=2): 80 筆 (33.33%)

資料分割完成：訓練集 192 筆, 驗證集 48 筆

模型將在 cuda 上運行

--- 開始 Finetune ---
Epoch 1/3: 100%|████████| 48/48 [02:15<00:00]
Epoch 2/3: 100%|████████| 48/48 [02:10<00:00]
Epoch 3/3: 100%|████████| 48/48 [02:08<00:00]

--- 訓練完成 ---
模型與 Tokenizer 已儲存至: ./my_finetuned_bert_kt_model
```

### 步驟 3: 檢視訓練結果

訓練完成後，可以使用 `view_training_history.py` 來檢視訓練曲線和摘要。該腳本會自動讀取 `results/` 目錄下最新的訓練結果。

```bash
python view_training_history.py
```

若要指定特定的結果目錄，請修改腳本中的 `TARGET_DIR` 變數。

### 步驟 4: 使用訓練好的模型

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 載入模型
model = BertForSequenceClassification.from_pretrained('./my_finetuned_bert_kt_model')
tokenizer = BertTokenizer.from_pretrained('./my_finetuned_bert_kt_model')

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
label_map = {0: "待加強", 1: "尚可", 2: "精熟"}
print(f"預測結果: {label_map[prediction]}")
```

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
