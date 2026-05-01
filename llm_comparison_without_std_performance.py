"""
LLM Comparison — Setting B（去除 [學生表現] leakage）
======================================================
完全複用 llm_comparison.py 的 predictor / evaluator / 視覺化邏輯，
唯一差別：載入 test set 後 in-memory 剝除 ［學生表現］：xxx 整行，
讓 LLM 與 Setting B RoBERTa 在「同樣無 leakage」的輸入條件下公平比較。

對應 Setting B RoBERTa multi-seed baseline (n=5):
    macro_f1 = 0.6518 ± 0.0461

執行流程（建議）：
    # Step 1: 先以小成本驗證（local Gemma + Claude）
    python llm_comparison_without_std_performance.py \
        --models gemma-3-4b-it claude-sonnet-4-5

    # Step 2: 驗證 OK 後跑全部 17 個模型
    python llm_comparison_without_std_performance.py --models all

備註：若也要包含 RoBERTa，請指定 Setting B 訓練的模型路徑：
    --models all roberta \
    --roberta-path results/roberta-chinese_setting_b_20260501_202000/final_model
"""

import sys
from datetime import datetime

import llm_comparison
from finetune_setting_b import PERF_LINE_PATTERN


_original_load = llm_comparison.load_test_dataset


def load_test_dataset_no_perf(splits_dir: str):
    """載入 test set 後 in-memory 剝除 ［學生表現］：xxx 整行。"""
    df = _original_load(splits_dir)
    if 'Short_Answer_Log' in df.columns:
        before_len = df['Short_Answer_Log'].str.len().sum()
        df['Short_Answer_Log'] = df['Short_Answer_Log'].apply(
            lambda s: PERF_LINE_PATTERN.sub('', s)
        )
        after_len = df['Short_Answer_Log'].str.len().sum()
        remaining = df['Short_Answer_Log'].str.contains('［學生表現］').sum()
        print()
        print("🔧 [Setting B] 已剝除 ［學生表現］ 整行")
        print(f"   Short_Answer_Log 字元總數: {before_len} → {after_len}  (-{before_len - after_len})")
        print(f"   剝除後殘留筆數: {remaining}  (應為 0)")
    return df


llm_comparison.load_test_dataset = load_test_dataset_no_perf


def main():
    if "--output-dir" not in sys.argv:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sys.argv.extend(["--output-dir", f"results/llm_comparison_setting_b_{timestamp}"])

    if "roberta" in sys.argv and "--roberta-path" not in sys.argv:
        print("\n⚠️  你選了 roberta 但沒指定 --roberta-path")
        print("   Setting B 公平比較應該用 Setting B 訓練的 RoBERTa，例如：")
        print("   --roberta-path results/roberta-chinese_setting_b_20260501_202000/final_model")
        print("   （否則 Setting A 模型遇到無 leakage 輸入，分數會比 Setting B 模型還低）\n")

    llm_comparison.main()


if __name__ == "__main__":
    main()
