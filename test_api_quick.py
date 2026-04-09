"""
快速 API 驗證腳本
================
每個模型用少量樣本 (3 筆) 確認 API 可正常呼叫。

用法:
    python test_api_quick.py --models all
    python test_api_quick.py --models gpt-4o gpt-4o-mini gpt-4.1 o4-mini
    python test_api_quick.py --models gemini-2.5-pro gemini-2.5-flash gemini-2.5-flash-lite
    python test_api_quick.py --models gemini-3.1-pro-preview gemini-3-flash-preview gemini-3.1-flash-lite-preview
    python test_api_quick.py --models claude-sonnet-4-5
    python test_api_quick.py --models llama-3.3-70b-versatile llama-3.1-8b-instant qwen/qwen3-32b
    python test_api_quick.py --models roberta
"""

import os
import sys
import time
import argparse
from dotenv import load_dotenv

load_dotenv()

# ─── 測試用固定樣本 ───────────────────────────────────────────────────────────
TEST_SAMPLES = [
    {
        "chapter": "第一章：資料庫基礎",
        "section": "主鍵與外來鍵",
        "short_answer_log": "Q: 什麼是主鍵？A: 主鍵是表格中唯一識別每筆記錄的欄位，不可重複也不可為空。",
        "expected": "精熟",
    },
    {
        "chapter": "第二章：SQL 查詢",
        "section": "JOIN 操作",
        "short_answer_log": "Q: 說明 INNER JOIN 的用途。A: 我不知道，好像是連接兩個表格。",
        "expected": "待加強",
    },
    {
        "chapter": "第三章：正規化",
        "section": "第一正規化",
        "short_answer_log": "Q: 什麼是第一正規化？A: 好像是把資料整理好，讓每個欄位只有一個值。",
        "expected": "尚可",
    },
]

VALID_LABELS = {"待加強", "尚可", "精熟"}

SYSTEM_PROMPT = """你是一個教育評估專家。根據以下學生的學習資料，判斷該學生對此知識點的掌握程度。
請只回答以下三個等級之一：待加強、尚可、精熟"""

USER_PROMPT_TEMPLATE = """章節：{chapter}
知識點：{section}
簡答題作答紀錄：
{short_answer_log}

請只回答一個詞：待加強、尚可、或精熟"""


def build_msg(sample):
    return USER_PROMPT_TEMPLATE.format(**sample)


def strip_think(text: str) -> str:
    """Strip <think>...</think> blocks emitted by reasoning models (Qwen3, DeepSeek etc)"""
    import re
    text = re.sub(r'<think>.*?</think>', '', str(text), flags=re.DOTALL).strip()
    # 處理不完整 think block（被 max_tokens 截斷，沒有結尾標籤）
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL).strip()
    return text


def print_result(idx, expected, response):
    response = strip_think(response)
    valid = response in VALID_LABELS
    match = "✅" if response == expected else ("⚠️" if valid else "❌")
    print(f"    [{idx+1}] 期望={expected}, 回應={response!r}  {match}")


# ─── 測試函數 ─────────────────────────────────────────────────────────────────

def test_roberta(roberta_path):
    print("\n🤖 [roberta] 測試中...")
    try:
        import torch, numpy as np
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        final = os.path.join(roberta_path, "final_model")
        if os.path.exists(final):
            roberta_path = final
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(roberta_path)
        mdl = AutoModelForSequenceClassification.from_pretrained(roberta_path).to(device).eval()
        ID2L = {0: "待加強", 1: "尚可", 2: "精熟"}

        for i, s in enumerate(TEST_SAMPLES):
            inp = tok(
                f"章節 : {s['chapter']}\n知識點 : {s['section']}\n學生掌握度 : [MASK]\n簡答題作答紀錄 :\n{s['short_answer_log']}\n",
                return_tensors="pt", max_length=512, padding="max_length", truncation=True
            )
            with torch.no_grad():
                out = mdl(input_ids=inp["input_ids"].to(device), attention_mask=inp["attention_mask"].to(device))
                pred = ID2L[int(torch.argmax(out.logits, dim=1).cpu())]
            print_result(i, s["expected"], pred)
        print("  ✓ 完成")
        return True
    except Exception as e:
        print(f"  ✗ 失敗: {e}")
        import traceback; traceback.print_exc()
        return False


def _test_openai_chat(model: str, use_reasoning=False):
    """通用 OpenAI chat API 測試"""
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY 未設定")
    client = OpenAI(api_key=api_key)

    for i, s in enumerate(TEST_SAMPLES):
        user_msg = build_msg(s)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]
        if use_reasoning:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                max_completion_tokens=200, reasoning_effort="low"
            )
        else:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                temperature=0, max_tokens=20
            )
        print_result(i, s["expected"], resp.choices[0].message.content.strip())
        time.sleep(0.3)


def test_openai(model_name):
    is_reasoning = model_name.startswith("o")
    print(f"\n🤖 [{model_name}] 測試中...{' (reasoning)' if is_reasoning else ''}")
    try:
        _test_openai_chat(model_name, use_reasoning=is_reasoning)
        print(f"  ✓ 完成")
        return True
    except Exception as e:
        print(f"  ✗ 失敗: {e}")
        import traceback; traceback.print_exc()
        return False


def _test_gemini(model: str):
    """通用 Gemini 測試"""
    from google import genai
    from google.genai import types
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY 未設定")
    client = genai.Client(api_key=api_key)

    sleep_time = 0.5 if "pro" in model else 0.3
    for i, s in enumerate(TEST_SAMPLES):
        full_prompt = f"{SYSTEM_PROMPT}\n\n{build_msg(s)}"
        resp = client.models.generate_content(
            model=model,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=2048,  # thinking 需要更多 token 配額
                thinking_config=types.ThinkingConfig(thinking_budget=1024),
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",       threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",       threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                ],
            ),
        )
        # 只取非思考 (thought=False) 的 parts
        text = None
        try:
            non_thought = [
                p.text for p in resp.candidates[0].content.parts
                if hasattr(p, "text") and p.text and not getattr(p, "thought", False)
            ]
            text = "".join(non_thought) or None
        except Exception:
            pass
        if text is None:
            try:
                text = resp.text
            except Exception:
                pass
        print_result(i, s["expected"], (text or "").strip())
        time.sleep(sleep_time)

        time.sleep(sleep_time)


def test_gemini(model_name):
    print(f"\n🤖 [{model_name}] 測試中...")
    try:
        _test_gemini(model_name)
        print(f"  ✓ 完成")
        return True
    except Exception as e:
        print(f"  ✗ 失敗: {e}")
        import traceback; traceback.print_exc()
        return False


def test_claude(model_api_name="claude-sonnet-4-5-20251022"):
    print(f"\n🤖 [claude-sonnet-4-5] 測試中...")
    try:
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY 未設定")
        client = anthropic.Anthropic(api_key=api_key)

        # 先列出可用模型確認
        try:
            models_page = client.models.list()
            available = [m.id for m in models_page.data]
            sonnet45 = [m for m in available if "sonnet-4-5" in m]
            if sonnet45:
                model_api_name = sorted(sonnet45)[-1]
            print(f"  → 使用模型: {model_api_name}")
        except Exception as e:
            print(f"  → 無法列出模型: {e}，使用預設: {model_api_name}")

        for i, s in enumerate(TEST_SAMPLES):
            resp = client.messages.create(
                model=model_api_name,
                max_tokens=20,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": build_msg(s)}]
            )
            print_result(i, s["expected"], resp.content[0].text.strip())
            time.sleep(0.5)

        print(f"  ✓ 完成")
        return model_api_name
    except Exception as e:
        print(f"  ✗ 失敗: {e}")
        import traceback; traceback.print_exc()
        return False


def _test_groq(model: str):
    """通用 Groq 測試"""
    from openai import OpenAI
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY 未設定")
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    for i, s in enumerate(TEST_SAMPLES):
        msg = build_msg(s)
        # Qwen3: 加 /no_think 關掉思考模式
        if "qwen3" in model:
            msg = msg + " /no_think"
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": msg},
            ],
            temperature=0,
            max_tokens=20,
        )
        print_result(i, s["expected"], resp.choices[0].message.content.strip())
        time.sleep(0.3)


def test_groq(model_name):
    print(f"\n🤖 [{model_name}] 測試中...")
    try:
        _test_groq(model_name)
        print(f"  ✓ 完成")
        return True
    except Exception as e:
        print(f"  ✗ 失敗: {e}")
        import traceback; traceback.print_exc()
        return False


# ─── 模型路由表 ───────────────────────────────────────────────────────────────
OPENAI_MODELS = {"gpt-4o", "gpt-4o-mini", "gpt-4.1", "o4-mini"}
GEMINI_MODELS = {
    "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
    "gemini-3.1-pro-preview", "gemini-3-flash-preview", "gemini-3.1-flash-lite-preview",
}
CLAUDE_MODELS = {"claude-sonnet-4-5"}
GROQ_MODELS   = {"llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen/qwen3-32b"}

ALL_MODELS = ["roberta"] + sorted(OPENAI_MODELS) + sorted(GEMINI_MODELS) + sorted(CLAUDE_MODELS) + sorted(GROQ_MODELS)


# ─── 主程式 ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="快速 API 驗證 (每模型 3 筆)")
    parser.add_argument(
        "--models", nargs="+", default=["all"],
        help=f"要測試的模型，或 all。可選: {ALL_MODELS}",
    )
    parser.add_argument(
        "--roberta-path", type=str,
        default="results/roberta-chinese_20260227_193714",
    )
    args = parser.parse_args()

    models = ALL_MODELS if "all" in args.models else args.models

    print("=" * 60)
    print("🧪 LLM API 快速驗證 (各 3 筆樣本)")
    print("=" * 60)
    print(f"  測試模型 ({len(models)}): {models}\n")

    results = {}
    for m in models:
        if m == "roberta":
            results[m] = test_roberta(args.roberta_path)
        elif m in OPENAI_MODELS:
            results[m] = test_openai(m)
        elif m in GEMINI_MODELS:
            results[m] = test_gemini(m)
        elif m in CLAUDE_MODELS:
            results[m] = test_claude()
        elif m in GROQ_MODELS:
            results[m] = test_groq(m)
        else:
            print(f"\n⚠️  未知模型: {m}，跳過")
            results[m] = False

    print("\n" + "=" * 60)
    print("📋 驗證摘要")
    print("=" * 60)
    for m, r in results.items():
        if r is False:
            status = "❌ 失敗"
        elif r is True:
            status = "✅ 通過"
        else:
            status = f"✅ 通過 (model: {r})"
        print(f"  {m:35s}: {status}")

    failed = [m for m, r in results.items() if r is False]
    if failed:
        print(f"\n⚠️  以下模型需要修復: {failed}")
        sys.exit(1)
    else:
        print("\n✅ 所有模型驗證通過！可以執行完整推論。")


if __name__ == "__main__":
    main()
