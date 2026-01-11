# RoBERTa Inference 與 LIME 視覺化 - 實施計畫

## Git Branch

- **Branch Name**: `feature/roberta-inference-lime`

## 概述

本計劃旨在為已訓練完成的 RoBERTa 模型添加兩個核心功能：

1. **Inference 推理腳本** - 使用訓練好的模型對新數據進行預測
2. **LIME 可解釋性視覺化** - 解釋模型為何做出特定預測，提高模型透明度
3. **LLM 關鍵字提取** - 使用 OpenAI API 提取語義關鍵字，提升 LIME 解釋品質

## 背景

根據現有代碼分析：
- ✅ 已完成 BERT 和 RoBERTa 模型訓練
- ✅ 最佳 RoBERTa 模型位於 `/home/culture/finetune_KT/results/roberta-chinese_20260111_034253/`
- ✅ 數據集使用 3 級掌握度分類：待加強 (0)、尚可 (1)、精熟 (2)
- ✅ 模型輸入格式為結構化文本（章節、知識點、簡答題紀錄、對話紀錄）

## 核心功能

### 1. RoBERTa Inference Engine (`inference.py`)

```python
from inference import RoBERTaInference

model = RoBERTaInference("results/roberta-chinese_20260111_034253/final_model/")
result = model.predict_single(sample)
```

- 支援單筆與批次推理
- CLI 與 Python API 兩種使用方式
- 輸出預測標籤、信心度與機率分佈

### 2. LIME Explainer (`lime_explainer.py`)

```python
from lime_explainer import LIMEExplainer

explainer = LIMEExplainer(model_path, use_llm_keywords=True)
exp, keywords = explainer.explain_prediction_with_keywords(sample, focus_on="student_answers")
explainer.generate_html_report(exp, "report.html", original_text=text, keywords=keywords)
```

**核心特色**：
- **LLM 關鍵字提取**：使用 OpenAI API (gpt-4o-mini) 自動提取語義關鍵字
- **針對性分析**：支援 `student_answers`、`student_questions`、`full_text` 等分析模式
- **學生表現標籤**：提取並分析 `Correct`、`Partially Correct`、`Incorrect` 標籤
- **自訂 HTML 報告**：完整原文顯示 + 關鍵字顏色高亮

### 3. 視覺化報告

報告包含：
1. **預測結果**：類別與機率分佈
2. **原始文本高亮**：
   - 🟢 綠色 = 正向影響（支持預測）
   - 🔴 紅色 = 負向影響（反對預測）
3. **特徵重要性表格**：按權重排序的關鍵詞列表

## 技術細節

### LLM 關鍵字提取

- **API**: OpenAI Chat Completions
- **模型**: gpt-4o-mini（可透過 `.env` 配置）
- **Prompt 設計**：提取學生回答中的關鍵詞 + 表現標籤
- **重疊處理**："Partially Correct" 優先於 "Correct"（長詞優先）

### 環境配置

```bash
# .env 文件
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=gpt-4o-mini  # 可選
```

## 文件結構

```
finetune_KT/
├── inference.py              # 推理引擎
├── lime_explainer.py         # LIME 解釋器（含 LLM 整合）
├── .env.example              # API 配置範例
├── requirements.txt          # 依賴套件
├── examples/
│   ├── inference_single.py   # 推理 API 範例
│   └── lime_explain_sample.py # LIME 使用範例
└── lime_reports/
    ├── explanation_answers_llm.html
    ├── explanation_questions_llm.html
    └── explanation_full_llm.html
```

## 驗證計畫

### 自動化測試

```bash
# 測試推理
export PYTHONPATH=.
python examples/inference_single.py

# 測試 LIME（需配置 .env）
python examples/lime_explain_sample.py
```

### 手動驗證

1. 檢查 HTML 報告的關鍵字高亮是否正確
2. 確認 "Partially Correct" 作為完整詞彙顯示
3. 驗證特徵權重是否合理

## 依賴套件

新增：
- `openai` - OpenAI API
- `python-dotenv` - 環境變數管理
- `lime` - LIME 可解釋性
- `jieba` - 中文分詞（fallback）
