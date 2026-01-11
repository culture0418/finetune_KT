# RoBERTa Inference & LIME Visualization Walkthrough

## Git Branch

- **Branch Name**: `feature/roberta-inference-lime`

## 1. 概述

本任務成功實現了 RoBERTa 模型的推理功能和 LIME 可解釋性視覺化模組，並整合了 **LLM 關鍵字提取** 功能，大幅提升解釋報告的語義品質。

主要目標是讓使用者能夠輕鬆載入訓練好的模型進行預測，並透過 LIME 深入了解模型為何做出特定判斷，特別是針對學生的回答和提問進行分析。

## 2. 核心功能實現

### 2.1 RoBERTa Inference Engine (`inference.py`)

- **類別設計**: `RoBERTaInference` 類別封裝了模型載入、數據格式化和預測邏輯。
- **模型支援**: 支援 BERT 和 RoBERTa 模型（自動檢測），使用 `AutoTokenizer` 確保兼容性。
- **介面支援**:
  - **Python API**: 方便整合到其他系統。
  - **CLI**: 提供互動式測試和批次 CSV 處理功能。
- **輸出**: 提供預測類別、Confidence 分數及各類別機率。

### 2.2 LIME Visualization Module (`lime_explainer.py`)

- **類別設計**: `LIMEExplainer` 類別負責生成局部解釋。
- **核心特色**: 
  - **LLM 關鍵字提取**: 使用 OpenAI API (gpt-4o-mini) 自動提取語義關鍵字
  - **針對性分析 (Targeted Analysis)**: 能夠從完整的對話紀錄中精確提取並分析
  - **學生表現標籤**: 提取 `Correct`、`Partially Correct`、`Incorrect` 作為獨立特徵
- **分析模式**:
  - `student_answers`: 僅分析學生的簡答題回答
  - `student_questions`: 僅分析學生的提問內容
  - `full_text`: 完整文本分析（結合所有關鍵字，無需重複 API 呼叫）
- **視覺化**: 生成互動式 HTML 報告，在完整原文中高亮顯示關鍵字

### 2.3 LLM 整合技術細節

- **API 配置**: 透過 `.env` 文件管理 API Key
- **Prompt 設計**: 針對學生回答和提問設計不同的提取策略
- **重疊處理**: 使用索引定位 + 區間合併，確保 "Partially Correct" 優先於 "Correct"
- **Token 合併**: 使用 `|||` 分隔符保持多詞短語完整性

## 3. 驗證結果

### 3.1 Inference 驗證

- **單筆推理**: 使用實際樣本測試，模型成功載入並輸出預測結果。
- **批次處理**: 測試了 CLI 的批次 CSV 處理功能，成功生成 `predictions.csv`。

### 3.2 LIME 驗證

執行了 `examples/lime_explain_sample.py`，成功生成了三份不同維度的解釋報告：

1. `explanation_answers_llm.html`: 聚焦於學生回答的影響
   - 關鍵字範例: `['電腦', '文字', '無法分辨', '人類表達方式', '智慧', '推理期', '知識期', '學習期', 'Correct', 'Partially Correct']`

2. `explanation_questions_llm.html`: 聚焦於學生提問的影響
   - 關鍵字範例: `['偏置', 'Bias', '激活函數', '差異', '什麼']`

3. `explanation_full_llm.html`: 完整文本的解釋
   - 結合上述所有關鍵字（無需額外 API 呼叫）

### 3.3 HTML 報告功能

- ✅ 完整原文顯示
- ✅ 關鍵字顏色高亮（綠色=正向，紅色=負向）
- ✅ 特徵重要性表格
- ✅ 預測機率分佈
- ✅ 懸停顯示權重值

## 4. 關鍵文件列表

| 文件 | 說明 |
|------|------|
| [inference.py](file:///home/culture/finetune_KT/inference.py) | 推理引擎核心代碼 |
| [lime_explainer.py](file:///home/culture/finetune_KT/lime_explainer.py) | LIME 解釋器核心代碼（含 LLM 整合） |
| [.env.example](file:///home/culture/finetune_KT/.env.example) | API 配置範例 |
| [examples/inference_single.py](file:///home/culture/finetune_KT/examples/inference_single.py) | API 使用範例 |
| [examples/lime_explain_sample.py](file:///home/culture/finetune_KT/examples/lime_explain_sample.py) | LIME 使用範例 |

## 5. 使用方式

### 環境配置

```bash
# 1. 安裝依賴
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 編輯 .env 填入 OPENAI_API_KEY
```

### 執行範例

```bash
cd finetune_KT
export PYTHONPATH=.

# 推理測試
python examples/inference_single.py

# LIME 解釋（需配置 .env）
python examples/lime_explain_sample.py
```

## 6. 後續建議

- 可以將這些功能整合到 Web 後端 API 中，為前端 Dashboard 提供即時診斷功能。
- HTML 報告也可以直接嵌入到網頁中顯示給教師或學生看。
- 可考慮增加更多 LLM 模型選項（如本地模型）以降低 API 成本。
