#!/usr/bin/env python3
"""
Token 長度分析腳本
用於分析 finetune_dataset_1132_v2.csv 中超過 512 tokens 的資料比例
"""

import pandas as pd
from transformers import BertTokenizer
import sys

def analyze_token_lengths(csv_path: str, model_name: str = "bert-base-chinese", threshold: int = 512):
    """
    分析資料集中的 token 長度分布
    
    Args:
        csv_path: CSV 檔案路徑
        model_name: Tokenizer 模型名稱
        threshold: Token 長度閾值
    """
    print(f"\n📊 正在分析 Token 長度...")
    print(f"資料集: {csv_path}")
    print(f"Tokenizer: {model_name}")
    print(f"閾值: {threshold} tokens\n")
    
    # 載入資料
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ 成功載入資料，共 {len(df)} 筆")
    except Exception as e:
        print(f"❌ 載入失敗: {e}")
        return
    
    # 顯示欄位資訊
    print(f"\n欄位列表: {list(df.columns)}")
    
    # 載入 Tokenizer
    try:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        print(f"✅ 成功載入 Tokenizer")
    except Exception as e:
        print(f"❌ Tokenizer 載入失敗: {e}")
        return
    
    # 處理 NaN 值
    df = df.fillna('')
    
    # 分析每筆資料的 token 長度
    lengths = []
    over_threshold_samples = []
    
    print(f"\n🔍 開始分析...")
    
    for idx, row in df.iterrows():
        # 格式化文本 - 根據新資料集的欄位結構
        # 使用 Short_Answer_Log 和 Dialog 欄位
        formatted_text = (
            f"章節 : {row.get('chapter', '')}\\n"
            f"知識點 : {row.get('section', '')}\\n"
            f"學生掌握度 : [MASK]\\n"
            f"簡答題作答紀錄 :\\n{row.get('Short_Answer_Log', '')}\\n"
            f"對話紀錄 :\\n{row.get('Dialog', '')}\\n"
        )
        
        # 計算 token 長度
        tokens = tokenizer.encode(formatted_text, add_special_tokens=True)
        seq_len = len(tokens)
        char_len = len(formatted_text)
        lengths.append(seq_len)
        
        # 記錄超過閾值的資料
        if seq_len > threshold:
            over_threshold_samples.append({
                'index': idx,
                'user_id': row.get('user_id', 'N/A'),
                'username': row.get('username', 'N/A'),
                'chapter': row.get('chapter', 'N/A'),
                'section': row.get('section', 'N/A'),
                'token_length': seq_len,
                'char_length': char_len
            })
    
    # 統計結果
    total_samples = len(df)
    over_threshold_count = len(over_threshold_samples)
    ratio = (over_threshold_count / total_samples) * 100 if total_samples > 0 else 0
    max_len = max(lengths) if lengths else 0
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    
    # 計算分位數
    lengths_sorted = sorted(lengths)
    p50 = lengths_sorted[len(lengths_sorted) // 2] if lengths else 0
    p75 = lengths_sorted[int(len(lengths_sorted) * 0.75)] if lengths else 0
    p90 = lengths_sorted[int(len(lengths_sorted) * 0.90)] if lengths else 0
    p95 = lengths_sorted[int(len(lengths_sorted) * 0.95)] if lengths else 0
    p99 = lengths_sorted[int(len(lengths_sorted) * 0.99)] if lengths else 0
    
    # 輸出分析報告
    print("\n" + "="*80)
    print("📊 Token 長度分析報告")
    print("="*80)
    print(f"總筆數: {total_samples:,}")
    print(f"最大長度: {max_len:,} tokens")
    print(f"平均長度: {avg_len:.1f} tokens")
    print(f"\n分位數統計:")
    print(f"  P50 (中位數): {p50:,} tokens")
    print(f"  P75: {p75:,} tokens")
    print(f"  P90: {p90:,} tokens")
    print(f"  P95: {p95:,} tokens")
    print(f"  P99: {p99:,} tokens")
    print(f"\n超過 {threshold} tokens 的資料:")
    print(f"  數量: {over_threshold_count:,} 筆")
    print(f"  比例: {ratio:.2f}%")
    print("="*80)
    
    # 顯示前 20 筆超過閾值的資料
    if over_threshold_samples:
        print(f"\n⚠️ 前 20 筆超過閾值的資料詳情:")
        print("-"*120)
        print(f"{'Index':<8} {'User ID':<12} {'Username':<15} {'Token長度':<12} {'字元長度':<12} {'章節':<30}")
        print("-"*120)
        
        for sample in over_threshold_samples[:20]:
            print(f"{sample['index']:<8} {str(sample['user_id']):<12} {sample['username']:<15} "
                  f"{sample['token_length']:<12} {sample['char_length']:<12} {sample['chapter']:<30}")
        
        if len(over_threshold_samples) > 20:
            print(f"\n... 還有 {len(over_threshold_samples) - 20} 筆超過閾值的資料未顯示")
    
    # 建議
    print("\n💡 建議:")
    if ratio > 10.0:
        print(f"  ⚠️ 有 {ratio:.2f}% 的資料長度超過 {threshold} tokens")
        print(f"  建議考慮使用長文本模型 (如 Longformer, BigBird) 或調整最大長度設定")
    elif ratio > 5.0:
        print(f"  ⚠️ 有 {ratio:.2f}% 的資料長度超過 {threshold} tokens")
        print(f"  可考慮調整最大長度或評估長文本模型的必要性")
    else:
        print(f"  ✅ 只有 {ratio:.2f}% 的資料超過 {threshold} tokens，可繼續使用 BERT")
    
    print("\n")
    
    return {
        'total': total_samples,
        'over_threshold': over_threshold_count,
        'ratio': ratio,
        'max_len': max_len,
        'avg_len': avg_len,
        'p50': p50,
        'p75': p75,
        'p90': p90,
        'p95': p95,
        'p99': p99
    }

if __name__ == "__main__":
    csv_path = "datasets/finetune_dataset_1132_v2.csv"
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    
    analyze_token_lengths(csv_path)
