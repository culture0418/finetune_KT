"""
論文與簡報結果生成腳本
======================
基於 llm_comparison.py 的結果，生成適合論文與簡報使用的精簡版圖表與深入錯誤分析。

輸出目錄結構:
paper_result/
├── enhanced_comparison_metrics.csv          # 增強版指標表格
├── focused_models/                          # 精簡版圖表（5 模型）
│   ├── overall_comparison.png
│   ├── per_class_heatmap.png
│   └── confusion_matrices/
│       ├── roberta.png
│       ├── gemini-3.1-pro-preview.png
│       └── gemini-2.5-flash.png
├── error_analysis/                          # 錯誤分析
│   ├── roberta_vs_llm.csv                  # RoBERTa 對但 LLM 錯
│   ├── all_models_wrong.csv                # 所有模型都錯
│   └── error_statistics.json               # 統計資訊
├── paper/                                   # 論文版（英文）
│   └── ...
└── presentation/                            # PPT 版（中英混合）
    └── ...

Usage:
    python generate_paper_results.py --source results/llm_comparison_<timestamp>
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import confusion_matrix

# ========================================
# 常數定義
# ========================================
LABEL_MAP = {"待加強": 0, "尚可": 1, "精熟": 2}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}
CLASS_NAMES_ZH = ["待加強", "尚可", "精熟"]
CLASS_NAMES_EN = ["Struggling", "Developing", "Mastered"]

# 代表模型（5 個）
FOCUSED_MODELS = [
    "roberta",
    "gemini-3.1-pro-preview",
    "gpt-4o-mini",
    "claude-sonnet-4-5",
    "qwen-qwen3-32b"
]

# 混淆矩陣對照組（3 個）
CONFUSION_MATRIX_MODELS = [
    "roberta",                      # 最佳
    "gemini-3.1-pro-preview",      # LLM 最佳
    "gemini-2.5-flash"             # 表現較差對照
]

# 模型類型分類
MODEL_TYPES = {
    "roberta": "Local (Finetuned)",
    "gpt-4o": "OpenAI",
    "gpt-4o-mini": "OpenAI",
    "gpt-4.1": "OpenAI",
    "o4-mini": "OpenAI",
    "gemini-2.5-pro": "Gemini 2.5",
    "gemini-2.5-flash": "Gemini 2.5",
    "gemini-2.5-flash-lite": "Gemini 2.5",
    "gemini-3.1-pro-preview": "Gemini 3.x",
    "gemini-3-flash-preview": "Gemini 3.x",
    "gemini-3.1-flash-lite-preview": "Gemini 3.x",
    "claude-sonnet-4-5": "Anthropic",
    "llama-3.3-70b-versatile": "Open Source (Groq)",
    "llama-3.1-8b-instant": "Open Source (Groq)",
    "qwen-qwen3-32b": "Open Source (Groq)",
}

# 層級分類（依 Accuracy）
def get_tier(accuracy: float) -> str:
    """依 Accuracy 分層級"""
    if accuracy >= 0.95:
        return "🥇 Best"
    elif accuracy >= 0.80:
        return "🥈 Excellent"
    elif accuracy >= 0.70:
        return "Good"
    elif accuracy >= 0.50:
        return "Fair"
    else:
        return "Poor"


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
            return True
    print("⚠ 未找到中文字體，圖表中文可能顯示為方塊")
    return False


# ========================================
# 任務 1: 增強版指標表格
# ========================================
def generate_enhanced_metrics(source_dir: str, output_dir: str):
    """
    生成增強版指標表格，加入模型類型與層級標記
    """
    print("\n" + "="*60)
    print("任務 1: 生成增強版指標表格")
    print("="*60)

    # 讀取原始指標
    metrics_file = os.path.join(source_dir, "comparison_metrics.csv")
    if not os.path.exists(metrics_file):
        raise FileNotFoundError(f"找不到指標檔案: {metrics_file}")

    df = pd.read_csv(metrics_file, encoding='utf-8-sig')

    # 加入模型類型
    df['model_type'] = df['model'].map(MODEL_TYPES)

    # 加入層級
    df['tier'] = df['accuracy'].apply(get_tier)

    # 依 Accuracy 排序
    df = df.sort_values('accuracy', ascending=False).reset_index(drop=True)

    # 調整欄位順序
    cols = ['model', 'model_type', 'tier', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall',
            'total_samples', 'valid_samples', 'invalid_count']
    per_class_cols = [c for c in df.columns if c not in cols]
    df = df[cols + per_class_cols]

    # 儲存
    output_file = os.path.join(output_dir, "enhanced_comparison_metrics.csv")
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✓ 增強版指標已存: {output_file}")

    # 顯示摘要
    print(f"\n模型總數: {len(df)}")
    print(f"層級分布:")
    print(df['tier'].value_counts().to_string())

    return df


# ========================================
# 任務 2: 精簡版圖表（5 模型）
# ========================================
def generate_focused_plots(source_dir: str, output_dir: str, metrics_df: pd.DataFrame, use_english: bool = False, dpi: int = 200):
    """
    生成精簡版圖表，只包含 5 個代表模型

    Args:
        source_dir: 原始結果目錄
        output_dir: 輸出目錄
        metrics_df: 指標 DataFrame
        use_english: 是否使用英文標籤
        dpi: 圖表解析度（論文版 300，PPT 版 150，預設 200）
    """
    task_name = "論文版" if use_english else "PPT 版" if dpi == 150 else "基礎版"
    print(f"\n  生成精簡版圖表（5 個代表模型，{task_name}）")

    focused_dir = os.path.join(output_dir, "focused_models")
    os.makedirs(focused_dir, exist_ok=True)

    # 篩選代表模型
    focused_df = metrics_df[metrics_df['model'].isin(FOCUSED_MODELS)].copy()
    focused_df = focused_df.set_index('model').loc[FOCUSED_MODELS].reset_index()

    print(f"選定模型: {FOCUSED_MODELS}")

    # 設定標籤
    class_names = CLASS_NAMES_EN if use_english else CLASS_NAMES_ZH

    # 2.1 Overall Metrics 對比圖
    _plot_focused_overall(focused_df, focused_dir, use_english, dpi)

    # 2.2 Per-Class F1 熱力圖
    _plot_focused_heatmap(focused_df, focused_dir, class_names, use_english, dpi)

    # 2.3 混淆矩陣（3 個模型）
    _plot_focused_confusion_matrices(source_dir, focused_dir, class_names, use_english, dpi)

    print(f"  ✓ 所有精簡版圖表已存至: {focused_dir}")


def _plot_focused_overall(df: pd.DataFrame, output_dir: str, use_english: bool, dpi: int = 200):
    """Overall Metrics 群組長條圖（5 模型）"""
    plot_metrics = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']

    if use_english:
        display_names = ['Accuracy', 'Macro-Precision', 'Macro-Recall', 'Macro-F1']
        title = 'Model Performance Comparison — Overall Metrics'
        xlabel = 'Model'
        ylabel = 'Score'
    else:
        display_names = ['Accuracy', 'Macro-Precision', 'Macro-Recall', 'Macro-F1']
        title = '模型效能比較 — Overall Metrics'
        xlabel = 'Model'
        ylabel = '分數'

    models = df['model'].tolist()
    n_models = len(models)
    x = np.arange(n_models)
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

    for i, (metric, display) in enumerate(zip(plot_metrics, display_names)):
        values = df[metric].tolist()
        bars = ax.bar(x + i * width, values, width, label=display, color=colors[i],
                     edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, fontsize=11, rotation=20, ha='right')
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, "overall_comparison.png")
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Overall 對比圖: {path}")


def _plot_focused_heatmap(df: pd.DataFrame, output_dir: str, class_names: list, use_english: bool, dpi: int = 200):
    """Per-Class F1 熱力圖（5 模型）"""
    models = df['model'].tolist()
    n_models = len(models)

    # 組成 data: shape = (n_class, n_models)
    data = np.array([
        [df.loc[df['model'] == m, f"{cls}_f1"].values[0] for m in models]
        for cls in CLASS_NAMES_ZH  # 使用中文名稱作為 key
    ])

    if use_english:
        title = 'Per-Class F1-Score Comparison'
        ylabel = 'Class'
        xlabel = 'Model'
    else:
        title = 'Per-Class F1-Score 比較'
        ylabel = '類別'
        xlabel = 'Model'

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(np.arange(n_models))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(models, rotation=30, ha='right', fontsize=11)
    ax.set_yticklabels(class_names, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=12, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=8)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)

    # 格子數值
    for i in range(len(class_names)):
        for j in range(n_models):
            val = data[i, j]
            color = 'white' if val > 0.6 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    color=color, fontsize=10, fontweight='bold')

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    path = os.path.join(output_dir, "per_class_heatmap.png")
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Per-Class 熱力圖: {path}")


def _plot_focused_confusion_matrices(source_dir: str, output_dir: str, class_names: list, use_english: bool, dpi: int = 200):
    """混淆矩陣（3 個對照模型）"""
    cm_dir = os.path.join(output_dir, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)

    if use_english:
        xlabel = 'Predicted Label'
        ylabel = 'True Label'
    else:
        xlabel = '預測標籤'
        ylabel = '真實標籤'

    pred_dir = os.path.join(source_dir, "predictions")

    for model_name in CONFUSION_MATRIX_MODELS:
        safe_name = model_name.replace("/", "-")
        pred_file = os.path.join(pred_dir, f"{safe_name}_predictions.csv")

        if not os.path.exists(pred_file):
            print(f"  ⚠ 找不到預測檔案: {pred_file}")
            continue

        df = pd.read_csv(pred_file, encoding='utf-8-sig')
        y_true = df['Mastery_Label'].tolist()
        y_pred = df['predicted'].tolist()

        # 過濾無效預測
        valid_labels = set(CLASS_NAMES_ZH)
        valid_mask = [p in valid_labels for p in y_pred]
        y_true = [y_true[i] for i in range(len(y_true)) if valid_mask[i]]
        y_pred = [y_pred[i] for i in range(len(y_pred)) if valid_mask[i]]

        if len(y_true) == 0:
            print(f"  ⚠ {model_name}: 無有效預測")
            continue

        cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES_ZH)

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title(f'{model_name}', fontsize=14, fontweight='bold', pad=12)

        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names, fontsize=12)
        ax.set_yticklabels(class_names, fontsize=12)
        ax.set_xlabel(xlabel, fontsize=13, labelpad=8)
        ax.set_ylabel(ylabel, fontsize=13, labelpad=8)

        thresh = cm.max() / 2. if cm.max() > 0 else 1
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=16, fontweight='bold')

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        path = os.path.join(cm_dir, f"{safe_name}.png")
        plt.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f"    ✓ 混淆矩陣: {path}")


# ========================================
# 任務 3: 錯誤分析
# ========================================
def generate_error_analysis(source_dir: str, output_dir: str, dataset_path: str):
    """
    深入錯誤分析
    1. RoBERTa 對但 LLM 錯的樣本
    2. 所有模型都錯的難題
    3. 按章節/知識點統計錯誤率
    """
    print("\n" + "="*60)
    print("任務 3: 錯誤分析")
    print("="*60)

    error_dir = os.path.join(output_dir, "error_analysis")
    os.makedirs(error_dir, exist_ok=True)

    # 載入原始資料集（需要章節/知識點資訊）
    dataset_df = pd.read_csv(dataset_path, encoding='utf-8-sig')

    # 載入所有模型預測
    pred_dir = os.path.join(source_dir, "predictions")
    predictions = {}

    for pred_file in sorted(Path(pred_dir).glob("*_predictions.csv")):
        model_key = pred_file.stem.replace("_predictions", "")
        pred_df = pd.read_csv(pred_file, encoding='utf-8-sig')
        predictions[model_key] = pred_df

    if not predictions:
        raise ValueError(f"找不到任何預測結果: {pred_dir}")

    print(f"載入 {len(predictions)} 個模型的預測結果")

    # 3.1 RoBERTa 對但 LLM 錯
    _analyze_roberta_vs_llm(predictions, error_dir)

    # 3.2 所有模型都錯
    _analyze_all_wrong(predictions, error_dir)

    # 3.3 按知識點統計
    _analyze_by_knowledge_point(predictions, error_dir)

    print(f"✓ 錯誤分析完成，結果存至: {error_dir}")


def _analyze_roberta_vs_llm(predictions: dict, output_dir: str):
    """RoBERTa 對但 LLM 錯的樣本"""
    print("\n  分析: RoBERTa 對但 LLM 錯...")

    if 'roberta' not in predictions:
        print("    ⚠ 找不到 RoBERTa 預測結果")
        return

    roberta_df = predictions['roberta']
    y_true = roberta_df['Mastery_Label'].tolist()
    roberta_pred = roberta_df['predicted'].tolist()

    # RoBERTa 答對的 index
    roberta_correct = set(i for i in range(len(y_true)) if roberta_pred[i] == y_true[i])

    # 統計每個 LLM 答錯的情況
    results = []

    for model_name, pred_df in predictions.items():
        if model_name == 'roberta':
            continue

        llm_pred = pred_df['predicted'].tolist()

        # RoBERTa 對但 LLM 錯
        roberta_right_llm_wrong = [
            i for i in roberta_correct
            if i < len(llm_pred) and llm_pred[i] != y_true[i]
        ]

        if roberta_right_llm_wrong:
            for idx in roberta_right_llm_wrong:
                # 組合學習資料作為 input
                row = pred_df.iloc[idx]
                input_text = (
                    f"章節：{row['chapter']}\n"
                    f"知識點：{row['section']}\n"
                    f"簡答題作答紀錄：\n{row['Short_Answer_Log']}"
                )

                results.append({
                    'model': model_name,
                    'sample_index': idx,
                    'chapter': row['chapter'],
                    'section': row['section'],
                    'input': input_text,
                    'true_label': y_true[idx],
                    'roberta_pred': roberta_pred[idx],
                    'llm_pred': llm_pred[idx],
                })

    if results:
        result_df = pd.DataFrame(results)
        output_file = os.path.join(output_dir, "roberta_vs_llm.csv")
        result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"    ✓ RoBERTa 對但 LLM 錯: {len(results)} 筆 → {output_file}")

        # 統計各模型錯誤數
        error_counts = result_df['model'].value_counts()
        print(f"    各模型錯誤數:")
        for model, count in error_counts.items():
            print(f"      {model}: {count}")
    else:
        print("    ℹ 沒有 RoBERTa 對但 LLM 錯的樣本")


def _analyze_all_wrong(predictions: dict, output_dir: str):
    """所有模型都答錯的難題"""
    print("\n  分析: 所有模型都答錯...")

    # 取第一個模型的標籤作為基準
    first_model = list(predictions.values())[0]
    y_true = first_model['Mastery_Label'].tolist()
    n_samples = len(y_true)

    # 統計每個樣本被多少模型答錯
    wrong_counts = [0] * n_samples

    for model_name, pred_df in predictions.items():
        pred = pred_df['predicted'].tolist()
        for i in range(min(len(pred), n_samples)):
            if pred[i] != y_true[i]:
                wrong_counts[i] += 1

    # 所有模型都錯的樣本
    all_wrong_indices = [i for i in range(n_samples) if wrong_counts[i] == len(predictions)]

    if all_wrong_indices:
        results = []
        for idx in all_wrong_indices:
            row = {
                'sample_index': idx,
                'chapter': first_model.iloc[idx]['chapter'],
                'section': first_model.iloc[idx]['section'],
                'true_label': y_true[idx],
            }
            # 加入各模型的預測
            for model_name, pred_df in predictions.items():
                row[f'{model_name}_pred'] = pred_df.iloc[idx]['predicted']
            results.append(row)

        result_df = pd.DataFrame(results)
        output_file = os.path.join(output_dir, "all_models_wrong.csv")
        result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"    ✓ 所有模型都錯: {len(all_wrong_indices)} 筆 → {output_file}")
    else:
        print("    ℹ 沒有所有模型都答錯的樣本")


def _analyze_by_knowledge_point(predictions: dict, output_dir: str):
    """按章節/知識點統計錯誤率"""
    print("\n  分析: 按知識點統計錯誤率...")

    # 取第一個模型的資料作為基準
    first_model = list(predictions.values())[0]
    y_true = first_model['Mastery_Label'].tolist()
    chapters = first_model['chapter'].tolist()
    sections = first_model['section'].tolist()

    # 建立知識點索引
    knowledge_points = {}
    for i, (ch, sec) in enumerate(zip(chapters, sections)):
        key = f"{ch} > {sec}"
        if key not in knowledge_points:
            knowledge_points[key] = []
        knowledge_points[key].append(i)

    # 統計每個知識點的錯誤率
    results = []

    for kp, indices in knowledge_points.items():
        ch, sec = kp.split(' > ')
        row = {
            'chapter': ch,
            'section': sec,
            'total_samples': len(indices),
        }

        # 計算每個模型在此知識點的錯誤率
        for model_name, pred_df in predictions.items():
            pred = pred_df['predicted'].tolist()
            errors = sum(1 for i in indices if i < len(pred) and pred[i] != y_true[i])
            error_rate = errors / len(indices) if indices else 0
            row[f'{model_name}_error_rate'] = error_rate

        results.append(row)

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('total_samples', ascending=False)

    output_file = os.path.join(output_dir, "error_by_knowledge_point.csv")
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"    ✓ 知識點錯誤率: {len(knowledge_points)} 個知識點 → {output_file}")

    # 統計資訊
    stats = {
        'total_knowledge_points': len(knowledge_points),
        'total_samples': len(y_true),
        'avg_samples_per_kp': len(y_true) / len(knowledge_points) if knowledge_points else 0,
    }

    stats_file = os.path.join(output_dir, "error_statistics.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"    ✓ 統計資訊: {stats_file}")


# ========================================
# 主程式
# ========================================
def main():
    parser = argparse.ArgumentParser(
        description="生成論文與簡報用的精簡版圖表與錯誤分析"
    )
    parser.add_argument(
        "--source", type=str, required=True,
        help="原始比較結果目錄 (例如 results/llm_comparison_YYYYMMDD_HHMMSS)"
    )
    parser.add_argument(
        "--dataset", type=str,
        default="datasets/splits/0227/test.csv",
        help="test set CSV 路徑（用於錯誤分析的章節/知識點資訊）"
    )
    parser.add_argument(
        "--output", type=str,
        default="paper_result",
        help="輸出目錄"
    )

    args = parser.parse_args()

    # 檢查來源目錄
    if not os.path.exists(args.source):
        raise FileNotFoundError(f"找不到來源目錄: {args.source}")

    # 建立輸出目錄
    os.makedirs(args.output, exist_ok=True)

    print("="*60)
    print("論文與簡報結果生成")
    print("="*60)
    print(f"來源: {args.source}")
    print(f"輸出: {args.output}")
    print()

    # 設定中文字體
    setup_chinese_font()

    # 任務 1: 增強版指標表格
    metrics_df = generate_enhanced_metrics(args.source, args.output)

    # 任務 2: 精簡版圖表（5 模型）- 基礎版
    print("\n" + "="*60)
    print("任務 2: 生成基礎版精簡圖表（5 個代表模型）")
    print("="*60)
    generate_focused_plots(args.source, args.output, metrics_df, use_english=False, dpi=200)

    # 任務 3: 錯誤分析
    if os.path.exists(args.dataset):
        generate_error_analysis(args.source, args.output, args.dataset)
    else:
        print(f"\n⚠ 找不到資料集 {args.dataset}，跳過錯誤分析")

    # 任務 4: 論文版圖表（全英文、300 DPI）
    paper_dir = os.path.join(args.output, "paper")
    os.makedirs(paper_dir, exist_ok=True)
    print("\n" + "="*60)
    print("任務 4: 生成論文版圖表（英文、高解析度 300 DPI）")
    print("="*60)
    generate_focused_plots(args.source, paper_dir, metrics_df, use_english=True, dpi=300)

    # 任務 5: PPT 版圖表（中英混合、150 DPI）
    ppt_dir = os.path.join(args.output, "presentation")
    os.makedirs(ppt_dir, exist_ok=True)
    print("\n" + "="*60)
    print("任務 5: 生成 PPT 版圖表（中英混合、標準解析度 150 DPI）")
    print("="*60)
    generate_focused_plots(args.source, ppt_dir, metrics_df, use_english=False, dpi=150)

    print("\n" + "="*60)
    print("✅ 所有結果生成完成")
    print("="*60)
    print(f"\n輸出目錄: {args.output}/")
    print("  ├── enhanced_comparison_metrics.csv")
    print("  ├── focused_models/            # 基礎版（中文）")
    print("  ├── paper/                     # 論文版（英文、300 DPI）")
    print("  │   ├── focused_models/")
    print("  │   │   ├── overall_comparison.png")
    print("  │   │   ├── per_class_heatmap.png")
    print("  │   │   └── confusion_matrices/")
    print("  ├── presentation/              # PPT 版（中英混合、150 DPI）")
    print("  │   └── focused_models/")
    print("  └── error_analysis/")
    print("      ├── roberta_vs_llm.csv")
    print("      ├── all_models_wrong.csv")
    print("      ├── error_by_knowledge_point.csv")
    print("      └── error_statistics.json")


if __name__ == "__main__":
    main()
