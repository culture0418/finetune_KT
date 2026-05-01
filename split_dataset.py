"""
Stratified train/val/test split for the finetune dataset.

預設產出 70/15/15 切分（random_state=42，依 Mastery_Label 做 stratify）。

輸出：
    <output-dir>/train.csv
    <output-dir>/val.csv
    <output-dir>/test.csv
    <output-dir>/split_info.json
"""

import argparse
import json
import os
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split


LABEL_MAP = {"待加強": 0, "尚可": 1, "精熟": 2}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}
REQUIRED_COLS = ['chapter', 'section', 'Short_Answer_Log', 'Mastery_Label']


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """載入原始資料集並清理（與 KTDataProcessor._load_and_clean 邏輯一致）"""
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    print(f"原始資料 '{csv_path}' 讀取成功，共 {len(df)} 筆")

    for col in REQUIRED_COLS:
        if col not in df.columns:
            print(f"警告：資料中缺少欄位 {col}，將自動建立並填入空值")
            df[col] = ''

    text_cols = [c for c in REQUIRED_COLS if c != 'Mastery_Label']
    for col in text_cols:
        df[col] = df[col].fillna('').astype(str)

    if df['Mastery_Label'].isnull().any():
        n_missing = df['Mastery_Label'].isnull().sum()
        print(f"警告：移除 {n_missing} 筆 'Mastery_Label' 為空的資料")
        df = df.dropna(subset=['Mastery_Label'])

    df['labels'] = df['Mastery_Label'].map(LABEL_MAP)

    if df['labels'].isnull().any():
        invalid_rows = df[df['labels'].isnull()]
        invalid_values = invalid_rows['Mastery_Label'].unique()
        print(f"警告：移除 {len(invalid_rows)} 筆無法識別的標籤值 (值={invalid_values})")
        df = df.dropna(subset=['labels']).copy()

    df['labels'] = df['labels'].astype(int)
    df = df.reset_index(drop=True)

    print(f"資料清理完成，剩餘 {len(df)} 筆有效資料")
    return df


def label_distribution(df: pd.DataFrame) -> dict:
    counts = df['labels'].value_counts().sort_index().to_dict()
    return {ID2LABEL[k]: int(v) for k, v in counts.items()}


def main():
    parser = argparse.ArgumentParser(description="切分 finetune dataset 為 train/val/test")
    parser.add_argument(
        "--csv", type=str,
        default="datasets/finetune_dataset_1142_v4_without_chat_0227.csv",
        help="原始資料集 CSV 路徑"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="datasets/splits/0227",
        help="切分結果輸出目錄"
    )
    parser.add_argument("--test-size", type=float, default=0.15,
                        help="test set 佔總資料的比例")
    parser.add_argument("--val-size", type=float, default=0.15,
                        help="val set 佔總資料的比例")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if args.test_size + args.val_size >= 1.0:
        raise ValueError("test_size + val_size 必須小於 1")

    df = load_and_clean(args.csv)
    total = len(df)

    # Stage 1: 切出 test
    trainval_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df['labels']
    )

    # Stage 2: 從 trainval 切出 val
    val_ratio_within_trainval = args.val_size / (1 - args.test_size)
    train_df, val_df = train_test_split(
        trainval_df,
        test_size=val_ratio_within_trainval,
        random_state=args.random_state,
        stratify=trainval_df['labels']
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"\n切分結果：")
    print(f"  Train: {len(train_df)} 筆 ({len(train_df)/total*100:.2f}%)")
    print(f"  Val:   {len(val_df)} 筆 ({len(val_df)/total*100:.2f}%)")
    print(f"  Test:  {len(test_df)} 筆 ({len(test_df)/total*100:.2f}%)")

    print("\n各 split 的標籤分佈：")
    print(f"  Train: {label_distribution(train_df)}")
    print(f"  Val:   {label_distribution(val_df)}")
    print(f"  Test:  {label_distribution(test_df)}")

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.csv")
    val_path = os.path.join(args.output_dir, "val.csv")
    test_path = os.path.join(args.output_dir, "test.csv")

    train_df.to_csv(train_path, index=False, encoding='utf-8-sig')
    val_df.to_csv(val_path, index=False, encoding='utf-8-sig')
    test_df.to_csv(test_path, index=False, encoding='utf-8-sig')

    info = {
        "source_csv": args.csv,
        "generated_at": datetime.now().isoformat(timespec='seconds'),
        "random_state": args.random_state,
        "stratify": "Mastery_Label",
        "split_ratio_target": {
            "train": round(1 - args.test_size - args.val_size, 4),
            "val": args.val_size,
            "test": args.test_size,
        },
        "split_ratio_actual": {
            "train": round(len(train_df) / total, 4),
            "val": round(len(val_df) / total, 4),
            "test": round(len(test_df) / total, 4),
        },
        "counts": {
            "total": total,
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "label_distribution": {
            "train": label_distribution(train_df),
            "val": label_distribution(val_df),
            "test": label_distribution(test_df),
        },
    }
    info_path = os.path.join(args.output_dir, "split_info.json")
    with open(info_path, "w", encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 切分完成，輸出至: {args.output_dir}/")
    print(f"   - train.csv ({len(train_df)} 筆)")
    print(f"   - val.csv ({len(val_df)} 筆)")
    print(f"   - test.csv ({len(test_df)} 筆)")
    print(f"   - split_info.json")


if __name__ == "__main__":
    main()
