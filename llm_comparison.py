"""
LLM Comparison Script: Finetuned RoBERTa vs LLM APIs
=====================================================
比較 finetuned RoBERTa 模型與多個 LLM API 在知識掌握度分類任務上的效能。
評測使用 split_dataset.py 切出的 test set（對 RoBERTa 訓練是 held-out）。

支援模型 (--models 的值直接對應 API model name):
  本地:    roberta
  OpenAI:  gpt-4o  gpt-4o-mini  gpt-4.1  o4-mini
  Gemini:  gemini-2.5-pro  gemini-2.5-flash  gemini-2.5-flash-lite
           gemini-3.1-pro  gemini-3-flash  gemini-3.1-flash-lite
  Gemma:   gemma-3-4b-it  gemma-3-12b-it  gemma-3-27b-it
  Claude:  claude-sonnet-4-5
  Groq:    llama-3.3-70b-versatile  llama-3.1-8b-instant  qwen/qwen3-32b

Usage:
    python llm_comparison.py --models all
    python llm_comparison.py --models gpt-4o gpt-4.1 gemini-2.5-pro claude-sonnet-4-5
    python llm_comparison.py --models all --skip-existing
    python llm_comparison.py --models all --output-dir results/my_run
    python llm_comparison.py --models gemma-3-27b-it --splits-dir datasets/splits/0227
"""

import os
import sys
import json
import time
import argparse
import re
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from dotenv import load_dotenv

load_dotenv()

# ========================================
# 中文字體設定
# ========================================
def setup_chinese_font():
    """設定 Matplotlib 中文字體"""
    font_paths = [
        "/usr/share/fonts/truetype/TaipeiSansTCBeta-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            fm.fontManager.addfont(fp)
            prop = fm.FontProperties(fname=fp)
            plt.rcParams['font.family'] = prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
            print(f"✓ 使用字體: {fp}")
            return
    print("⚠ 未找到中文字體，圖表中文可能顯示為方塊")

setup_chinese_font()

# ========================================
# 常數
# ========================================
LABEL_MAP = {"待加強": 0, "尚可": 1, "精熟": 2}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}
VALID_LABELS = set(LABEL_MAP.keys())

SYSTEM_PROMPT = """你是一個教育評估專家。根據以下學生的學習資料，判斷該學生對此知識點的掌握程度。
請只回答以下三個等級之一：待加強、尚可、精熟"""

USER_PROMPT_TEMPLATE = """章節：{chapter}
知識點：{section}
簡答題作答紀錄：
{short_answer_log}

請只回答一個詞：待加強、尚可、或精熟"""

# ========================================
# 資料載入
# ========================================
def load_test_dataset(splits_dir: str) -> pd.DataFrame:
    """
    載入由 split_dataset.py 預先切好的 test set。
    test set 對 RoBERTa 訓練是 held-out，可用來公平比較 RoBERTa vs LLM。
    """
    test_path = os.path.join(splits_dir, "test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"找不到 {test_path}\n請先執行: python split_dataset.py --output-dir {splits_dir}"
        )

    df = pd.read_csv(test_path, encoding='utf-8-sig')

    text_cols = ['chapter', 'section', 'Short_Answer_Log']
    for col in text_cols:
        if col not in df.columns:
            df[col] = ''
        df[col] = df[col].fillna('').astype(str)

    if 'labels' not in df.columns:
        df['labels'] = df['Mastery_Label'].map(LABEL_MAP)
        df = df.dropna(subset=['labels']).copy()
    df['labels'] = df['labels'].astype(int)
    df = df.reset_index(drop=True)

    print(f"✓ 載入 test set: {test_path}")
    print(f"  共 {len(df)} 筆")
    print(f"  Test 標籤分佈: {dict(df['Mastery_Label'].value_counts())}")

    return df


def parse_llm_response(response) -> str:
    """
    解析 LLM 回應，提取分類標籤。
    - 支援 None 輸入
    - 自動處理 <think>...</think> (Qwen3/DeepSeek 思考模式漏出)
    """
    if response is None:
        return "無法判斷"

    response = str(response).strip()

    # 去除 <think>...</think> 區塊（Qwen3 / DeepSeek thinking mode）
    import re
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    # 處理不完整的 think block（被 max_tokens 截斷，沒有結尾標籤）
    response = re.sub(r'<think>.*$', '', response, flags=re.DOTALL).strip()

    # 直接匹配
    if response in VALID_LABELS:
        return response

    # 在回應中搜尋標籤
    for label in ["待加強", "精熟", "尚可"]:  # 先匹配較明確的
        if label in response:
            return label

    return "無法判斷"


# ========================================
# Predictor 基類
# ========================================
class BasePredictor(ABC):
    """所有 Predictor 的抽象基類"""
    
    def __init__(self, name: str, model_id: str):
        self.name = name
        self.model_id = model_id
    
    @abstractmethod
    def predict_single(self, chapter: str, section: str, short_answer_log: str) -> str:
        """預測單筆資料，回傳標籤字串 (待加強/尚可/精熟)"""
        pass
    
    def predict_batch(self, data_df: pd.DataFrame, output_dir: str, skip_existing: bool = False) -> pd.DataFrame:
        """
        批次預測，支援中斷續跑。
        回傳包含預測結果的 DataFrame。
        """
        safe_name = self.name.replace("/", "-")
        pred_file = os.path.join(output_dir, f"{safe_name}_predictions.csv")
        
        # 檢查是否有已存在的結果
        if skip_existing and os.path.exists(pred_file):
            print(f"  ⏭ 跳過 {self.name} (已有結果: {pred_file})")
            return pd.read_csv(pred_file, encoding='utf-8-sig')
        
        # 檢查是否有未完成的中間結果
        partial_file = os.path.join(output_dir, f"{safe_name}_partial.csv")
        start_idx = 0
        predictions = []
        
        if os.path.exists(partial_file):
            partial_df = pd.read_csv(partial_file, encoding='utf-8-sig')
            start_idx = len(partial_df)
            predictions = partial_df['predicted'].tolist()
            print(f"  ↩ 從第 {start_idx} 筆續跑 ({self.name})")
        
        total = len(data_df)
        print(f"  ▶ {self.name} 推論中 ({start_idx}/{total})...")

        for idx in range(start_idx, total):
            row = data_df.iloc[idx]
            try:
                pred = self.predict_single(
                    chapter=str(row['chapter']),
                    section=str(row['section']),
                    short_answer_log=str(row['Short_Answer_Log'])
                )
                predictions.append(pred)
            except Exception as e:
                print(f"    ⚠ 第 {idx} 筆錯誤: {e}")
                predictions.append("無法判斷")
            
            # 每 10 筆存一次中間結果
            if (idx + 1) % 10 == 0 or idx == total - 1:
                partial_save = data_df.iloc[:len(predictions)].copy()
                partial_save['predicted'] = predictions
                partial_save.to_csv(partial_file, index=False, encoding='utf-8-sig')
                print(f"    進度: {len(predictions)}/{total}")
        
        # 儲存最終結果
        result_df = data_df.copy()
        result_df['predicted'] = predictions
        result_df.to_csv(pred_file, index=False, encoding='utf-8-sig')
        
        # 清除中間檔案
        if os.path.exists(partial_file):
            os.remove(partial_file)
        
        print(f"  ✓ {self.name} 完成，結果存至 {pred_file}")
        return result_df


# ========================================
# RoBERTa Predictor
# ========================================
class RoBERTaPredictor(BasePredictor):
    """使用 finetuned RoBERTa 模型進行推論"""
    
    def __init__(self, model_path: str):
        super().__init__("roberta", "RoBERTa-finetuned")
        self.model_path = model_path
        self._load_model()
    
    def _load_model(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        final_model_path = os.path.join(self.model_path, "final_model")
        if os.path.exists(final_model_path):
            self.model_path = final_model_path
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        self.torch = torch
        print(f"  ✓ RoBERTa 模型載入完成 (device: {self.device})")
    
    def predict_single(self, chapter: str, section: str, short_answer_log: str) -> str:
        formatted_text = (
            f"章節 : {chapter}\n"
            f"知識點 : {section}\n"
            f"學生掌握度 : [MASK]\n"
            f"簡答題作答紀錄 :\n{short_answer_log}\n"
        )
        
        encoding = self.tokenizer(
            formatted_text,
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with self.torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = self.torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
            pred_id = int(np.argmax(probs))
        
        return ID2LABEL[pred_id]


# ========================================
# OpenAI Predictors (GPT-4o / 4o-mini / 4.1 / o4-mini)
# ========================================
class _OpenAIBase(BasePredictor):
    """OpenAI chat-completions API 基類"""

    def __init__(self, model: str):
        super().__init__(model, model)
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 未設定")
        self.client = OpenAI(api_key=api_key)
        print(f"  ✓ OpenAI 初始化完成 (model: {model})")

    def _call(self, messages: list, **kwargs) -> str:
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_tokens=20,
                    **kwargs,
                )
                return parse_llm_response(resp.choices[0].message.content)
            except Exception as e:
                if attempt < 2:
                    wait = 2 ** (attempt + 1)
                    print(f"    retry in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    raise


class GPT4oPredictor(_OpenAIBase):
    """OpenAI GPT-4o"""
    def __init__(self):
        super().__init__("gpt-4o")

    def predict_single(self, chapter, section, short_answer_log):
        user_msg = USER_PROMPT_TEMPLATE.format(
            chapter=chapter, section=section, short_answer_log=short_answer_log)
        return self._call(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user",   "content": user_msg}],
            temperature=0,
        )


class GPT4oMiniPredictor(_OpenAIBase):
    """OpenAI GPT-4o-mini"""
    def __init__(self):
        super().__init__("gpt-4o-mini")

    def predict_single(self, chapter, section, short_answer_log):
        user_msg = USER_PROMPT_TEMPLATE.format(
            chapter=chapter, section=section, short_answer_log=short_answer_log)
        return self._call(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user",   "content": user_msg}],
            temperature=0,
        )


class GPT41Predictor(_OpenAIBase):
    """OpenAI GPT-4.1"""
    def __init__(self):
        super().__init__("gpt-4.1")

    def predict_single(self, chapter, section, short_answer_log):
        user_msg = USER_PROMPT_TEMPLATE.format(
            chapter=chapter, section=section, short_answer_log=short_answer_log)
        return self._call(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user",   "content": user_msg}],
            temperature=0,
        )


class O4MiniPredictor(_OpenAIBase):
    """OpenAI o4-mini — 推理模型，不支援 temperature"""
    def __init__(self):
        super().__init__("o4-mini")

    def predict_single(self, chapter, section, short_answer_log):
        user_msg = USER_PROMPT_TEMPLATE.format(
            chapter=chapter, section=section, short_answer_log=short_answer_log)
        # o4-mini 不支援 temperature，改用 reasoning_effort
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_msg},
                    ],
                    max_completion_tokens=200,
                    reasoning_effort="low",  # low 將思考時間降到最低
                )
                return parse_llm_response(resp.choices[0].message.content)
            except Exception as e:
                if attempt < 2:
                    wait = 2 ** (attempt + 1)
                    print(f"    retry in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    raise



# ========================================
# Gemini Predictors (2.5 Pro / Flash / Flash-Lite / 3.x)
# ========================================
def _gemini_client():
    from google import genai
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY 未設定")
    return genai.Client(api_key=api_key)


class _GeminiBase(BasePredictor):
    """Gemini API 基類 - 子類直接傳入 model name"""

    # 不同模型的延遲設定（避免 rate limit）
    SLEEP_TIME = {
        "gemini-2.5-pro": 0.5,
        "gemini-3.1-pro": 0.5,
        # Flash 系列論量寬鬆
    }

    def __init__(self, model: str):
        super().__init__(model, model)
        self.client = _gemini_client()
        self._sleep = self.SLEEP_TIME.get(model, 0.2)
        print(f"  ✓ Gemini 初始化完成 (model: {model})")

    def predict_single(self, chapter, section, short_answer_log):
        from google.genai import types
        user_msg = USER_PROMPT_TEMPLATE.format(
            chapter=chapter, section=section, short_answer_log=short_answer_log)
        full_prompt = f"{SYSTEM_PROMPT}\n\n{user_msg}"

        for attempt in range(3):
            try:
                resp = self.client.models.generate_content(
                    model=self.model_id,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0,
                        max_output_tokens=1024,
                        # thinking_budget: gemini-2.5-pro 必須啟用 thinking，設 1024 最小值
                        # 從 response parts 中過濾掉 thought=True 的部分
                        thinking_config=types.ThinkingConfig(thinking_budget=1024),
                        safety_settings=[
                            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",        threshold="BLOCK_NONE"),
                            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",        threshold="BLOCK_NONE"),
                            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT",  threshold="BLOCK_NONE"),
                            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT",  threshold="BLOCK_NONE"),
                        ]
                    )
                )
                text = None
                try:
                    # 只取非思考 (thought=False) 的 parts
                    non_thought = [
                        p.text for p in resp.candidates[0].content.parts
                        if hasattr(p, 'text') and p.text and not getattr(p, 'thought', False)
                    ]
                    text = ''.join(non_thought) or None
                except Exception:
                    pass
                if text is None:
                    try:
                        text = resp.text
                    except Exception:
                        pass
                time.sleep(self._sleep)
                return parse_llm_response(text)

            except Exception as e:
                if attempt < 2:
                    wait = 2 ** (attempt + 1)
                    print(f"    retry in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    raise


class GeminiProPredictor(_GeminiBase):
    def __init__(self): super().__init__("gemini-2.5-pro")

class GeminiFlashPredictor(_GeminiBase):
    def __init__(self): super().__init__("gemini-2.5-flash")

class GeminiFlashLitePredictor(_GeminiBase):
    def __init__(self): super().__init__("gemini-2.5-flash-lite")

class Gemini3ProPredictor(_GeminiBase):
    def __init__(self): super().__init__("gemini-3.1-pro-preview")

class Gemini3FlashPredictor(_GeminiBase):
    def __init__(self): super().__init__("gemini-3-flash-preview")

class Gemini3FlashLitePredictor(_GeminiBase):
    def __init__(self): super().__init__("gemini-3.1-flash-lite-preview")


# ========================================
# Gemma 3 Predictors (Open-weight, via Google AI Studio API)
# ========================================
class _GemmaBase(BasePredictor):
    """Gemma 3 走 Google AI Studio 端點，但不支援 thinking_config / safety_settings。"""

    SLEEP_TIME = {
        "gemma-3-27b-it": 0.4,
        "gemma-3-12b-it": 0.3,
        "gemma-3-4b-it": 0.2,
    }

    def __init__(self, model: str):
        super().__init__(model, model)
        self.client = _gemini_client()
        self._sleep = self.SLEEP_TIME.get(model, 0.3)
        print(f"  ✓ Gemma 初始化完成 (model: {model})")

    def predict_single(self, chapter, section, short_answer_log):
        from google.genai import types
        user_msg = USER_PROMPT_TEMPLATE.format(
            chapter=chapter, section=section, short_answer_log=short_answer_log)
        # Gemma 不支援 system role，把 SYSTEM_PROMPT 併進使用者訊息
        full_prompt = f"{SYSTEM_PROMPT}\n\n{user_msg}"

        for attempt in range(3):
            try:
                resp = self.client.models.generate_content(
                    model=self.model_id,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0,
                        max_output_tokens=128,
                    )
                )
                text = None
                try:
                    text = resp.text
                except Exception:
                    pass
                if text is None:
                    try:
                        text = ''.join(
                            p.text for p in resp.candidates[0].content.parts
                            if hasattr(p, 'text') and p.text
                        ) or None
                    except Exception:
                        pass
                time.sleep(self._sleep)
                return parse_llm_response(text)

            except Exception as e:
                if attempt < 2:
                    wait = 2 ** (attempt + 1)
                    print(f"    retry in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    raise


class Gemma3_4BPredictor(_GemmaBase):
    def __init__(self): super().__init__("gemma-3-4b-it")

class Gemma3_12BPredictor(_GemmaBase):
    def __init__(self): super().__init__("gemma-3-12b-it")

class Gemma3_27BPredictor(_GemmaBase):
    def __init__(self): super().__init__("gemma-3-27b-it")



# ========================================
# Gemini Flash Predictor (2.0) — REMOVED (replaced by GeminiFlashPredictor 2.5)
# ========================================


# ========================================
# Claude Predictor
# ========================================
class ClaudePredictor(BasePredictor):
    """Anthropic Claude Sonnet 4.5"""

    def __init__(self):
        model = "claude-sonnet-4-5-20250929"
        super().__init__("claude-sonnet-4-5", model)

        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY 未設定")
        self.client = anthropic.Anthropic(api_key=api_key)
        print(f"  ✓ Claude 初始化完成 (model: {model})")

    def predict_single(self, chapter, section, short_answer_log):
        user_msg = USER_PROMPT_TEMPLATE.format(
            chapter=chapter, section=section, short_answer_log=short_answer_log)
        for attempt in range(3):
            try:
                resp = self.client.messages.create(
                    model=self.model_id,
                    max_tokens=20,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}]
                )
                return parse_llm_response(resp.content[0].text)
            except Exception as e:
                if attempt < 2:
                    wait = 2 ** (attempt + 1)
                    print(f"    retry in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    raise


# ========================================
# Groq Predictors (Llama 70B / 8B / Qwen3-32B)
# ========================================
class _GroqBase(BasePredictor):
    """Groq OpenAI-compatible API 基類"""

    def __init__(self, model: str):
        super().__init__(model, model)
        from openai import OpenAI
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY 未設定")
        self.client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        print(f"  ✓ Groq 初始化完成 (model: {model})")

    def predict_single(self, chapter, section, short_answer_log):
        user_msg = USER_PROMPT_TEMPLATE.format(
            chapter=chapter, section=section, short_answer_log=short_answer_log)
        # Qwen3: 加 /no_think 指令關抓思考模式
        if "qwen3" in self.model_id:
            user_msg = user_msg + " /no_think"
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_msg},
                    ],
                    temperature=0,
                    max_tokens=20,
                )
                return parse_llm_response(resp.choices[0].message.content)
            except Exception as e:
                if attempt < 2:
                    wait = 2 ** (attempt + 1)
                    print(f"    retry in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    raise


class Llama70bPredictor(_GroqBase):
    def __init__(self): super().__init__("llama-3.3-70b-versatile")

class Llama8bPredictor(_GroqBase):
    def __init__(self): super().__init__("llama-3.1-8b-instant")

class Qwen3Predictor(_GroqBase):
    def __init__(self): super().__init__("qwen/qwen3-32b")




# ========================================
# Comparison Evaluator
# ========================================
class ComparisonEvaluator:
    """計算評估指標並生成視覺化"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.class_names = ["待加強", "尚可", "精熟"]
    
    def compute_metrics(self, y_true: list, y_pred: list, model_name: str) -> dict:
        """計算單一模型的所有評估指標"""
        # 過濾掉無法判斷的
        valid_mask = [p in VALID_LABELS for p in y_pred]
        y_true_f = [y_true[i] for i in range(len(y_true)) if valid_mask[i]]
        y_pred_f = [y_pred[i] for i in range(len(y_pred)) if valid_mask[i]]
        
        invalid_count = len(y_true) - len(y_true_f)
        if invalid_count > 0:
            print(f"  ⚠ {model_name}: {invalid_count}/{len(y_true)} 筆無法判斷")
        
        if len(y_true_f) == 0:
            return {"model": model_name, "error": "全部無法判斷"}
        
        metrics = {
            "model": model_name,
            "total_samples": len(y_true),
            "valid_samples": len(y_true_f),
            "invalid_count": invalid_count,
            "accuracy": accuracy_score(y_true_f, y_pred_f),
            "macro_precision": precision_score(y_true_f, y_pred_f, labels=self.class_names, average='macro', zero_division=0),
            "macro_recall": recall_score(y_true_f, y_pred_f, labels=self.class_names, average='macro', zero_division=0),
            "macro_f1": f1_score(y_true_f, y_pred_f, labels=self.class_names, average='macro', zero_division=0),
        }
        
        # Per-class metrics
        for cls in self.class_names:
            y_true_bin = [1 if y == cls else 0 for y in y_true_f]
            y_pred_bin = [1 if y == cls else 0 for y in y_pred_f]
            
            metrics[f"{cls}_precision"] = precision_score(y_true_bin, y_pred_bin, zero_division=0)
            metrics[f"{cls}_recall"] = recall_score(y_true_bin, y_pred_bin, zero_division=0)
            metrics[f"{cls}_f1"] = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        
        return metrics
    
    def evaluate_all(self, results: dict, y_true: list) -> pd.DataFrame:
        """
        評估所有模型。
        results: {model_name: [predictions]}
        y_true: 真實標籤
        """
        all_metrics = []
        for model_name, y_pred in results.items():
            m = self.compute_metrics(y_true, y_pred, model_name)
            all_metrics.append(m)
        
        df = pd.DataFrame(all_metrics)
        csv_path = os.path.join(self.output_dir, "comparison_metrics.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n✓ Metrics 總表存至: {csv_path}")
        
        return df
    
    def plot_overall_comparison(self, metrics_df: pd.DataFrame):
        """群組長條圖：各模型的 Overall 指標比較"""
        plot_metrics = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']
        display_names = ['Accuracy', 'Macro-Precision', 'Macro-Recall', 'Macro-F1']
        
        models = metrics_df['model'].tolist()
        n_models = len(models)
        x = np.arange(n_models)
        width = 0.18
        
        # 依模型數量動態調整寬度，避免文字重疊
        fig_width = max(18, n_models * 1.8)
        fig, ax = plt.subplots(figsize=(fig_width, 9))
        
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
        
        for i, (metric, display) in enumerate(zip(plot_metrics, display_names)):
            values = metrics_df[metric].tolist()
            bars = ax.bar(x + i * width, values, width, label=display, color=colors[i], edgecolor='white', linewidth=0.5)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
        
        ax.set_xlabel('模型', fontsize=13)
        ax.set_ylabel('分數', fontsize=13)
        ax.set_title('模型效能比較 — Overall Metrics', fontsize=18, fontweight='bold', pad=20)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models, fontsize=10, rotation=15, ha='right')
        ax.set_ylim(0, 1.2)
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        path = os.path.join(self.output_dir, "overall_comparison.png")
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"✓ 圖表已存: {path}")
    
    def plot_per_class_heatmap(self, metrics_df: pd.DataFrame, heatmap_dir: str):
        """熱力圖：依指標拆分 3 張，rows=class，columns=模型"""
        metric_types = ['precision', 'recall', 'f1']
        display_types = ['Precision', 'Recall', 'F1-Score']

        models = metrics_df['model'].tolist()
        n_models = len(models)

        for mtype, display in zip(metric_types, display_types):
            # 組成 data: shape = (n_class, n_models)
            data = np.array([
                [metrics_df.loc[metrics_df['model'] == m, f"{cls}_{mtype}"].values[0]
                 for m in models]
                for cls in self.class_names
            ])

            # 依模型數量動態調整寬度，避免 column label 重疊
            fig_width = max(18, n_models * 2.0)
            fig, ax = plt.subplots(figsize=(fig_width, 6))

            im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

            ax.set_xticks(np.arange(n_models))
            ax.set_yticks(np.arange(len(self.class_names)))
            # 模型名稱旋轉 40° 避免重疊
            ax.set_xticklabels(models, rotation=40, ha='right', fontsize=11)
            ax.set_yticklabels(self.class_names, fontsize=13)
            ax.set_title(f'Per-Class {display} — 各模型比較', fontsize=15, fontweight='bold', pad=12)

            # 格子數值
            for i in range(len(self.class_names)):
                for j in range(n_models):
                    val = data[i, j]
                    color = 'white' if val > 0.6 else 'black'
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                            color=color, fontsize=10, fontweight='bold')

            plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
            # bottom 留空間給旋轉後的 x 軸標籤
            plt.tight_layout(rect=[0, 0.05, 1, 1])
            path = os.path.join(heatmap_dir, f"heatmap_{mtype}.png")
            plt.savefig(path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"  ✓ 熱力圖已存: {path}")

        print(f"✓ 所有熱力圖已存至: {heatmap_dir}")

    def plot_confusion_matrices(self, results: dict, y_true: list, cm_dir: str):
        """每個模型獨立輸出混淆矩陣 PNG 到 confusion_matrices/ 子資料夾"""
        for model_name, y_pred in results.items():
            safe_name = model_name.replace('/', '-')

            # 過濾無效預測
            valid_mask = [p in VALID_LABELS for p in y_pred]
            y_true_f = [y_true[i] for i in range(len(y_true)) if valid_mask[i]]
            y_pred_f = [y_pred[i] for i in range(len(y_pred)) if valid_mask[i]]

            # 增大圖片尺寸，避免標籤被截
            fig, ax = plt.subplots(figsize=(7, 6))

            if len(y_true_f) == 0:
                ax.set_title(f'{model_name}\n(無有效預測)', fontsize=13)
            else:
                cm = confusion_matrix(y_true_f, y_pred_f, labels=self.class_names)

                im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
                ax.set_title(f'{model_name}', fontsize=14, fontweight='bold', pad=12)

                tick_marks = np.arange(len(self.class_names))
                ax.set_xticks(tick_marks)
                ax.set_yticks(tick_marks)
                # x 軸標籤不旋轉，字體夠大且只有 3 個標籤
                ax.set_xticklabels(self.class_names, fontsize=12)
                ax.set_yticklabels(self.class_names, fontsize=12)
                ax.set_xlabel('預測標籤', fontsize=13, labelpad=8)
                ax.set_ylabel('真實標籤', fontsize=13, labelpad=8)

                thresh = cm.max() / 2. if cm.max() > 0 else 1
                for i in range(len(self.class_names)):
                    for j in range(len(self.class_names)):
                        ax.text(j, i, format(cm[i, j], 'd'),
                                ha="center", va="center",
                                color="white" if cm[i, j] > thresh else "black",
                                fontsize=16, fontweight='bold')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            plt.tight_layout()
            path = os.path.join(cm_dir, f"confusion_matrix_{safe_name}.png")
            plt.savefig(path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"  ✓ 混淆矩陣已存: {path}")

        print(f"✓ 所有混淆矩陣已存至: {cm_dir}")

    
    def plot_per_class_f1_bars(self, metrics_df: pd.DataFrame):
        """各模型 Per-Class F1 水平分組長條圖"""
        models = metrics_df['model'].tolist()
        n_models = len(models)
        
        # 依模型數量動態調整高度，避免文字重疊
        fig_height = max(10, n_models * 1.8)
        fig, ax = plt.subplots(figsize=(14, fig_height))
        
        colors = ['#E91E63', '#FF9800', '#4CAF50']  # 待加強, 尚可, 精熟
        bar_height = 0.25
        y = np.arange(n_models)
        
        for i, cls in enumerate(self.class_names):
            col = f"{cls}_f1"
            values = metrics_df[col].tolist()
            bars = ax.barh(y + i * bar_height, values, bar_height,
                           label=cls, color=colors[i], edgecolor='white', linewidth=0.5)
            for bar, val in zip(bars, values):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2.,
                        f'{val:.3f}', ha='left', va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('F1-Score', fontsize=13)
        ax.set_title('各模型 Per-Class F1-Score 比較', fontsize=18, fontweight='bold', pad=20)
        ax.set_yticks(y + bar_height)
        ax.set_yticklabels(models, fontsize=12)
        ax.set_xlim(0, 1.18)
        ax.legend(loc='lower right', fontsize=12)
        ax.grid(axis='x', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.invert_yaxis()
        
        plt.tight_layout()
        path = os.path.join(self.output_dir, "per_class_f1_comparison.png")
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"✓ 圖表已存: {path}")
    
    def generate_all_plots(self, metrics_df: pd.DataFrame, results: dict, y_true: list):
        """生成所有視覺化圖表"""
        print("\n📊 生成視覺化圖表...")
        
        # 建立子資料夾
        cm_dir = os.path.join(self.output_dir, "confusion_matrices")
        heatmap_dir = os.path.join(self.output_dir, "heatmaps")
        os.makedirs(cm_dir, exist_ok=True)
        os.makedirs(heatmap_dir, exist_ok=True)
        
        self.plot_overall_comparison(metrics_df)
        self.plot_per_class_heatmap(metrics_df, heatmap_dir)
        self.plot_confusion_matrices(results, y_true, cm_dir)
        self.plot_per_class_f1_bars(metrics_df)


# ========================================
# 主程式
# ========================================
MODEL_REGISTRY = {
    # ── Local ──────────────────────────────────────────────
    "roberta": {
        "class": None,  # handled specially (needs roberta_path)
        "description": "Finetuned RoBERTa (local)"
    },
    # ── OpenAI ─────────────────────────────────────────────
    "gpt-4o": {
        "class": GPT4oPredictor,
        "description": "OpenAI GPT-4o"
    },
    "gpt-4o-mini": {
        "class": GPT4oMiniPredictor,
        "description": "OpenAI GPT-4o-mini"
    },
    "gpt-4.1": {
        "class": GPT41Predictor,
        "description": "OpenAI GPT-4.1"
    },
    "o4-mini": {
        "class": O4MiniPredictor,
        "description": "OpenAI o4-mini (reasoning)"
    },
    # ── Google Gemini 2.5 ──────────────────────────────────
    "gemini-2.5-pro": {
        "class": GeminiProPredictor,
        "description": "Google Gemini 2.5 Pro"
    },
    "gemini-2.5-flash": {
        "class": GeminiFlashPredictor,
        "description": "Google Gemini 2.5 Flash"
    },
    "gemini-2.5-flash-lite": {
        "class": GeminiFlashLitePredictor,
        "description": "Google Gemini 2.5 Flash-Lite"
    },
    # ── Google Gemini 3.x (Preview) ────────────────────────
    "gemini-3.1-pro-preview": {
        "class": Gemini3ProPredictor,
        "description": "Google Gemini 3.1 Pro (Preview)"
    },
    "gemini-3-flash-preview": {
        "class": Gemini3FlashPredictor,
        "description": "Google Gemini 3 Flash (Preview)"
    },
    "gemini-3.1-flash-lite-preview": {
        "class": Gemini3FlashLitePredictor,
        "description": "Google Gemini 3.1 Flash-Lite (Preview)"
    },
    # ── Google Gemma 3 (Open Weights) ──────────────────────
    "gemma-3-4b-it": {
        "class": Gemma3_4BPredictor,
        "description": "Google Gemma 3 4B Instruct (open-weight)"
    },
    "gemma-3-12b-it": {
        "class": Gemma3_12BPredictor,
        "description": "Google Gemma 3 12B Instruct (open-weight)"
    },
    "gemma-3-27b-it": {
        "class": Gemma3_27BPredictor,
        "description": "Google Gemma 3 27B Instruct (open-weight)"
    },
    # ── Anthropic ──────────────────────────────────────────
    "claude-sonnet-4-5": {
        "class": ClaudePredictor,
        "description": "Anthropic Claude Sonnet 4.5"
    },
    # ── Groq (Open Source) ─────────────────────────────────
    "llama-3.3-70b-versatile": {
        "class": Llama70bPredictor,
        "description": "Meta Llama 3.3 70B (via Groq)"
    },
    "llama-3.1-8b-instant": {
        "class": Llama8bPredictor,
        "description": "Meta Llama 3.1 8B (via Groq)"
    },
    "qwen/qwen3-32b": {
        "class": Qwen3Predictor,
        "description": "Qwen3 32B (via Groq)"
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="比較 Finetuned RoBERTa 與 LLM APIs 的知識掌握度分類效能"
    )
    parser.add_argument(
        "--models", nargs="+", default=["all"],
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        help="要比較的模型 (預設: all)"
    )
    parser.add_argument(
        "--splits-dir", type=str,
        default="datasets/splits/0227",
        help="預切分目錄 (需包含 test.csv，由 split_dataset.py 產生)"
    )
    parser.add_argument(
        "--roberta-path", type=str,
        default="results/roberta-chinese_20260227_193714",
        help="RoBERTa 模型路徑"
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="跳過已有預測結果的模型"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="自訂輸出目錄 (預設: results/llm_comparison_YYYYMMDD_HHMMSS)"
    )
    
    args = parser.parse_args()
    
    # 決定要跑的模型
    if "all" in args.models:
        model_names = list(MODEL_REGISTRY.keys())
    else:
        model_names = args.models
    
    # 建立輸出目錄
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/llm_comparison_{timestamp}"
    
    pred_dir = os.path.join(output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    
    print("=" * 60)
    print("🔬 Finetuned RoBERTa vs LLM APIs 效能比較")
    print("=" * 60)
    print(f"  輸出目錄: {output_dir}")
    print(f"  比較模型: {model_names}")
    print(f"  Splits 目錄: {args.splits_dir}")
    print()

    # 載入 test set (RoBERTa 訓練未見過的 held-out split)
    test_df = load_test_dataset(args.splits_dir)
    y_true = test_df['Mastery_Label'].tolist()
    
    # 逐一執行各模型的推論
    for model_name in model_names:
        print(f"\n{'─' * 40}")
        print(f"📌 {model_name}: {MODEL_REGISTRY[model_name]['description']}")
        print(f"{'─' * 40}")
        
        try:
            if model_name == "roberta":
                predictor = RoBERTaPredictor(args.roberta_path)
            else:
                predictor = MODEL_REGISTRY[model_name]["class"]()

            
            predictor.predict_batch(test_df, pred_dir, args.skip_existing)
            
        except Exception as e:
            print(f"  ✗ {model_name} 失敗: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # === 從 predictions/ 目錄收集所有已完成的預測結果 ===
    print(f"\n{'=' * 60}")
    print("📊 收集所有已完成的預測結果並計算評估指標")
    print(f"{'=' * 60}")
    
    results = {}
    for pred_file in sorted(Path(pred_dir).glob("*_predictions.csv")):
        model_key = pred_file.stem.replace("_predictions", "")
        pred_df = pd.read_csv(pred_file, encoding='utf-8-sig')
        if 'predicted' in pred_df.columns:
            results[model_key] = pred_df['predicted'].tolist()
            print(f"  ✓ 載入 {model_key} 的預測結果 ({len(pred_df)} 筆)")
    
    if not results:
        print("\n✗ predictions/ 目錄中沒有任何預測結果")
        sys.exit(1)
    
    evaluator = ComparisonEvaluator(output_dir)
    metrics_df = evaluator.evaluate_all(results, y_true)
    
    # 顯示結果
    print("\n" + metrics_df.to_string(index=False))
    
    # 生成圖表 (包含所有已完成的模型)
    evaluator.generate_all_plots(metrics_df, results, y_true)
    
    print(f"\n{'=' * 60}")
    print(f"✅ 完成！所有結果存放於: {output_dir}")
    print(f"  包含 {len(results)} 個模型: {list(results.keys())}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
