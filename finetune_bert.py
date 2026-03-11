import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import os
import sys
from datetime import datetime
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    BertConfig,
    BertTokenizer,
    BertForSequenceClassification,
    RobertaConfig,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import torch.nn as nn
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances

# ========================================
# 模型配置字典
# ========================================
MODEL_CONFIGS = {
    "bert-base-chinese": {
        "config_class": BertConfig,
        "tokenizer_class": BertTokenizer,
        "model_class": BertForSequenceClassification,
        "max_token_len": 512,
        "description": "BERT Base Chinese"
    },
    "hfl/chinese-roberta-wwm-ext": {
        "config_class": BertConfig,  # 注意：HFL 的 Chinese RoBERTa 仍使用 BERT 架構
        "tokenizer_class": BertTokenizer,
        "model_class": BertForSequenceClassification,
        "max_token_len": 512,
        "description": "RoBERTa WWM Chinese (Whole Word Masking)"
    }
}

class TrainingVisualizer:
    """
    功用：負責將訓練過程中的 Log 轉化為圖表與 CSV 報表。
    """
    def __init__(self, output_dir: str, model_name: str = "BERT"):
        self.output_dir = output_dir
        self.model_name = model_name  # 保存模型名稱用於圖表標題
        self._ensure_dir_exists(self.output_dir)
        matplotlib.use('Agg')         # 設定非互動式後端，避免在 Server 上報錯
        self._set_chinese_font()       # 設定中文字體
    
    @staticmethod
    def _set_chinese_font():
        """設定 Matplotlib 使用 Taipei Sans TC Beta 字體以正確顯示中文"""
        font_path = "/home/culture/.local/share/fonts/TaipeiSansTCBeta-Bold.ttf"
        
        if not os.path.exists(font_path):
            print(f"⚠️ 找不到字體檔案：{font_path}")
            print("⚠️ 將使用系統預設字體，中文可能顯示為方塊")
            return None
        
        prop = fm.FontProperties(fname=font_path)
        font_name = prop.get_name() if prop.get_name() else 'sans-serif'
        plt.rcParams['font.family'] = font_name
        plt.rcParams['font.sans-serif'] = [font_name]
        plt.rcParams['axes.unicode_minus'] = False
        
        print(f"✅ Matplotlib 已設定字體：{font_name} ({font_path})")
        return prop
    
    @staticmethod
    def _ensure_dir_exists(path: str):
        """確保目標資料夾及其所有父資料夾存在"""
        os.makedirs(path, exist_ok=True)
        print(f"✓ 資料夾已確認存在: {path}")

    def plot(self, log_history: list):
        """
        繪製訓練曲線並儲存
        
        Args:
            log_history (list): Trainer.state.log_history (List of dicts)
        """
        print("\n📊 正在繪製訓練曲線...")
        
        # 1. 資料提取
        epochs = []
        train_losses = []
        eval_losses = []
        eval_accs = []

        for log in log_history:
            epoch = log.get('epoch', 0)
            
            # 收集訓練 loss (通常是每個 logging_steps 紀錄一次)
            if 'loss' in log:
                train_losses.append({
                    'epoch': epoch,
                    'loss': log['loss']
                })
            
            # 收集驗證 loss 和 accuracy (通常是每個 epoch 紀錄一次)
            if 'eval_loss' in log:
                epochs.append(epoch)
                eval_losses.append(log['eval_loss'])
                eval_accs.append(log.get('eval_accuracy', 0))

        if not epochs and not train_losses:
            print("⚠️ 警告：Log history 為空，無法繪圖。")
            return

        # 2. 繪製主要圖表 (Loss 和 Overall Accuracy)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'{self.model_name} Knowledge Tracing - Training History', fontsize=16, fontweight='bold')

        # 子圖 1: Validation Loss
        if eval_losses:
            ax1.plot(epochs, eval_losses, marker='o', label='Validation Loss', color='red', linewidth=2)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Validation Loss Curve')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # 子圖 2: Validation Accuracy
        if eval_accs:
            ax2.plot(epochs, eval_accs, marker='o', label='Validation Accuracy', color='green', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Validation Accuracy Curve')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 標記最佳點
            max_acc = max(eval_accs)
            max_acc_idx = eval_accs.index(max_acc)
            ax2.annotate(f'Best: {max_acc:.4f}', 
                         xy=(epochs[max_acc_idx], max_acc),
                         xytext=(10, -15), textcoords='offset points',
                         fontsize=10, color='green',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

        # 子圖 3: Training Loss
        if train_losses:
            t_epochs = [x['epoch'] for x in train_losses]
            t_values = [x['loss'] for x in train_losses]
            ax3.plot(t_epochs, t_values, label='Training Loss', color='blue', linewidth=1, alpha=0.7)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Loss')
            ax3.set_title('Training Loss Curve (Step-by-Step)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        
        # 3. 儲存主要圖表
        img_path = os.path.join(self.output_dir, 'training_metrics_visualization.png')
        plt.savefig(img_path, dpi=300)
        print(f"✅ 圖表已儲存至: {img_path}")
        plt.close(fig)

        # 4. 繪製每個類別的指標圖表
        self._plot_per_class_metrics(log_history, epochs)
        
        # 5. 繪製最終指標 Heatmap
        self._plot_final_metrics_heatmap(log_history)

        # 6. 儲存 CSV 摘要
        if eval_accs:
            # 動態收集所有 eval_ 開頭的指標
            csv_data = {
                'Epoch': epochs,
                'Eval_Loss': eval_losses,
                'Eval_Accuracy': eval_accs
            }
            
            # 提取每個類別的指標 - 更新為 3 個等級
            label_names = ["待加強", "尚可", "精熟"]
            metric_types = ["precision", "recall", "f1", "accuracy"]
            
            # 為每個類別和指標類型建立欄位
            for label_name in label_names:
                for metric_type in metric_types:
                    metric_key = f"eval_{label_name}_{metric_type}"
                    values = []
                    for log in log_history:
                        if 'eval_loss' in log:  # 只在 eval 的 log 中提取
                            values.append(log.get(metric_key, 0.0))
                    if values:
                        csv_data[f"{label_name}_{metric_type}"] = values
            
            df = pd.DataFrame(csv_data)
            csv_path = os.path.join(self.output_dir, 'training_metrics_summary.csv')
            df.to_csv(csv_path, index=False)
            print(f"✅ 摘要已儲存至: {csv_path}")

    def _plot_per_class_metrics(self, log_history: list, epochs: list):
        """
        繪製每個類別的 precision, recall, f1, accuracy 曲線
        
        Args:
            log_history (list): Trainer.state.log_history
            epochs (list): Epoch 列表
        """
        if not epochs:
            return
        
        # 3 個掌握度等級：待加強、尚可、精熟
        label_names = ["待加強", "尚可", "精熟"]
        metric_types = ["precision", "recall", "f1", "accuracy"]
        
        # 提取每個類別的所有指標
        class_metrics = {label: {metric: [] for metric in metric_types} for label in label_names}
        
        for log in log_history:
            if 'eval_loss' in log:  # 只處理 eval 的 log
                for label_name in label_names:
                    for metric_type in metric_types:
                        metric_key = f"eval_{label_name}_{metric_type}"
                        value = log.get(metric_key, 0.0)
                        class_metrics[label_name][metric_type].append(value)
        
        # 檢查是否有數據
        has_data = any(len(class_metrics[label][metric]) > 0 
                       for label in label_names 
                       for metric in metric_types)
        
        if not has_data:
            print("⚠️ 警告：沒有找到每個類別的指標數據，跳過繪製。")
            return
        
        # 定義顏色方案
        colors = {
            "待加強": "#e74c3c",  # 紅色
            "尚可": "#f39c12",    # 橙色
            # 藍色
            "精熟": "#2ecc71"     # 綠色
        }
        
        # 創建 2x2 子圖 (4個指標)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Per-Class Metrics - Training Progress', fontsize=16, fontweight='bold')
        
        # 將 axes 展平以便迭代
        axes_flat = axes.flatten()
        
        # 為每個指標類型繪製圖表
        for idx, metric_type in enumerate(metric_types):
            ax = axes_flat[idx]
            
            # 為每個類別繪製曲線
            for label_name in label_names:
                values = class_metrics[label_name][metric_type]
                if values:
                    ax.plot(epochs, values, 
                           marker='o', 
                           label=label_name, 
                           color=colors[label_name], 
                           linewidth=2,
                           markersize=6)
            
            # 設置圖表屬性
            metric_display_name = metric_type.upper() if metric_type == 'f1' else metric_type.capitalize()
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel(metric_display_name, fontsize=11)
            ax.set_title(f'{metric_display_name} by Class', fontsize=13, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.05])  # 設置 y 軸範圍 0-1
        
        plt.tight_layout()
        
        # 儲存圖表
        per_class_img_path = os.path.join(self.output_dir, 'per_class_metrics_visualization.png')
        plt.savefig(per_class_img_path, dpi=300)
        print(f"✅ 每個類別指標圖表已儲存至: {per_class_img_path}")
        plt.close(fig)
    
    def _plot_final_metrics_heatmap(self, log_history: list):
        """
        繪製混淆矩陣 Heatmap
        顯示真實標籤 vs 預測標籤，以了解哪些類別容易被誤判
        
        Args:
            log_history (list): Trainer.state.log_history
        """
        # 注意：此方法需要在訓練後重新評估以獲取預測結果
        # 由於 log_history 中沒有完整的預測數據，我們需要另外處理
        # 這個方法將在 BertKTFinetuner.run_finetuning() 中被調用
        print("⚠️ 混淆矩陣將在訓練完成後生成")
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """
        繪製混淆矩陣 Heatmap
        
        Args:
            y_true: 真實標籤
            y_pred: 預測標籤
            class_names: 類別名稱列表
        """
        from sklearn.metrics import confusion_matrix
        import numpy as np
        
        # 計算混淆矩陣
        cm = confusion_matrix(y_true, y_pred)
        
        # 創建圖表
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 繪製 heatmap
        im = ax.imshow(cm, cmap='YlOrRd', aspect='auto')
        
        # 設置 ticks
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, fontsize=12)
        ax.set_yticklabels(class_names, fontsize=12)
        
        # 設置軸標籤
        ax.set_xlabel('預測標籤 (Predicted Label)', fontsize=14, fontweight='bold')
        ax.set_ylabel('真實標籤 (True Label)', fontsize=14, fontweight='bold')
        ax.set_title('混淆矩陣 (Confusion Matrix)', fontsize=16, fontweight='bold', pad=20)
        
        # 在每個格子中顯示數值和百分比
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                # 計算百分比（相對於該真實類別的總數）
                percentage = (cm[i, j] / cm[i].sum() * 100) if cm[i].sum() > 0 else 0
                
                # 根據背景顏色選擇文字顏色
                text_color = "white" if cm[i, j] > cm.max() / 2 else "black"
                
                # 顯示數量和百分比
                text = ax.text(j, i, f'{cm[i, j]}\n({percentage:.1f}%)',
                             ha="center", va="center", color=text_color, 
                             fontsize=13, fontweight='bold')
        
        # 添加顏色條
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('樣本數量', rotation=270, labelpad=20, fontsize=12)
        
        # 添加網格線
        ax.set_xticks(np.arange(len(class_names)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(class_names)) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linestyle='-', linewidth=3)
        
        plt.tight_layout()
        
        # 儲存圖表
        confusion_matrix_path = os.path.join(self.output_dir, 'confusion_matrix_heatmap.png')
        plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
        print(f"✅ 混淆矩陣 Heatmap 已儲存至: {confusion_matrix_path}")
        plt.close(fig)


class KTDataProcessor:
    """
    功用：專門處理 finetune_dataset_1132_v2.csv 檔案。
    負責載入、清理、轉換標籤（3 個掌握度等級：待加強、尚可、精熟），並分割資料集。
    """
    def __init__(self, csv_path: str):
        """
        初始化物件，傳入 CSV 檔案的路徑。
        
        Args:
            csv_path (str): 'finetune_dataset_1132_v2.csv' 的路徑
        """
        self.csv_path = csv_path
        # 定義欄位和標籤 - 更新為新資料集結構
        self.required_cols = ['chapter', 'section', 'Short_Answer_Log', 'Mastery_Label']
        # 3 個掌握度等級：待加強、尚可、精熟
        self.label_map = {"待加強": 0, "尚可": 1, "精熟": 2}
        self.id2label = {v: k for k, v in self.label_map.items()}
        self.num_labels = len(self.label_map)  # num_labels = 3
        
        # 這些變數將在 prepare_data() 後被賦值
        self.train_df = None
        self.val_df = None

    def _load_and_clean(self) -> pd.DataFrame:
        """
        [內部方法] 載入 CSV 並進行基本清理。
        """
        try:
            # 建議：加上 encoding='utf-8-sig' 可以處理 Excel 存出的 CSV 常見的 BOM 問題
            # keep_default_na=True 是預設值，會自動識別 'NA', 'NaN' 等
            df = pd.read_csv(self.csv_path, encoding='utf-8-sig')
            print(f"原始資料 '{self.csv_path}' 讀取成功，共 {len(df)} 筆")
        except FileNotFoundError:
            print(f"錯誤：找不到檔案 {self.csv_path}")
            raise
        except pd.errors.EmptyDataError:
            print(f"錯誤：檔案 {self.csv_path} 內容為空")
            raise

        # 1. 補齊缺少欄位
        for col in self.required_cols:
            if col not in df.columns:
                print(f"警告：資料中缺少欄位 {col}，將自動建立並填入空值")
                df[col] = ''

        # 2. 清理特徵欄位 (非 Label)
        # 目標：填補 NaN 為空字串，並強制轉為 String 型別 (避免數字被當成 float)
        text_cols = [c for c in self.required_cols if c != 'Mastery_Label']
        for col in text_cols:
            # fillna('') 把 NaN 變成空字串
            # astype(str) 確保即使 CSV 裡是數字 123，也會變成字串 "123"
            df[col] = df[col].fillna('').astype(str)

        # 3. 清理標籤欄位 (Mastery_Label - 新資料集欄位名稱)
        # 步驟 A: 先把 Mastery_Label 本身是 NaN/空值的去掉
        if df['Mastery_Label'].isnull().any():
            n_missing = df['Mastery_Label'].isnull().sum()
            print(f"警告：移除 {n_missing} 筆 'Mastery_Label' 為空的資料")
            df = df.dropna(subset=['Mastery_Label'])

        # 步驟 B: 進行 Mapping
        df['labels'] = df['Mastery_Label'].map(self.label_map)

        # 步驟 C: 檢查 Mapping 後是否產生 NaN (代表出現了字典裡沒有的標籤)
        if df['labels'].isnull().any():
            invalid_rows = df[df['labels'].isnull()]
            invalid_values = invalid_rows['Mastery_Label'].unique()
            print(f"警告：移除 {len(invalid_rows)} 筆無法識別的標籤值")
            print(f"      無法識別的值包含: {invalid_values}")
            df = df.dropna(subset=['labels']).copy() # 這裡 copy 很重要，避免 SettingWithCopyWarning

        # 4. 最終整理
        # 轉成整數型別
        df['labels'] = df['labels'].astype(int)
        
        # 重置索引 (Reset Index)，讓索引從 0 開始連續，避免後續分割或 batch 出錯
        df = df.reset_index(drop=True)

        print(f"資料清理完成，剩餘 {len(df)} 筆有效資料")
        return df

    def prepare_data(self, test_size=0.2, random_state=42):
        """
        執行資料準備的主要流程：載入、清理、分割。
        使用 stratify 確保訓練集和驗證集中，三種標籤（待加強、尚可、精熟）的比例相同。
        """
        df = self._load_and_clean()
        
        # 顯示原始資料的標籤分佈
        print("\n原始資料標籤分佈：")
        label_counts = df['labels'].value_counts().sort_index()
        for label_id, count in label_counts.items():
            label_name = self.id2label[label_id]
            percentage = (count / len(df)) * 100
            print(f"  {label_name} (label={label_id}): {count} 筆 ({percentage:.2f}%)")
        
        
        # 分割訓練集和驗證集，使用 stratify 確保標籤分佈均衡
        self.train_df, self.val_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['labels']  # 關鍵：依據標籤進行分層抽樣
        )
        
        print(f"\n資料分割完成：訓練集 {len(self.train_df)} 筆, 驗證集 {len(self.val_df)} 筆")
        
        # 顯示訓練集的標籤分佈
        print("\n訓練集標籤分佈：")
        train_label_counts = self.train_df['labels'].value_counts().sort_index()
        for label_id, count in train_label_counts.items():
            label_name = self.id2label[label_id]
            percentage = (count / len(self.train_df)) * 100
            print(f"  {label_name} (label={label_id}): {count} 筆 ({percentage:.2f}%)")
        
        # 顯示驗證集的標籤分佈
        print("\n驗證集標籤分佈：")
        val_label_counts = self.val_df['labels'].value_counts().sort_index()
        for label_id, count in val_label_counts.items():
            label_name = self.id2label[label_id]
            percentage = (count / len(self.val_df)) * 100
            print(f"  {label_name} (label={label_id}): {count} 筆 ({percentage:.2f}%)")

    def get_dataframes(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """獲取已分割的訓練和驗證 DataFrame"""
        if self.train_df is None or self.val_df is None:
            raise ValueError("請先呼叫 .prepare_data() 來處理資料")
        return self.train_df, self.val_df

    def analyze_token_lengths(self, model_name: str, threshold: int = 512):
        """
        預先掃描資料集，統計 Token 長度分佈。
        
        Args:
            model_name (str): 使用的 Tokenizer 模型名稱
            threshold (int): 長度閾值 (預設 512)
            
        Returns:
            tuple: (超過閾值的比例 %, 最大長度)
        """
        # 載入 Tokenizer
        try:
            tokenizer = BertTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f"警告: 無法載入 Tokenizer ({e})，跳過長度分析。")
            return 0.0, 0

        print(f"\n📊 正在分析 Token 長度 (使用 {model_name})...")
        
        # 合併資料進行分析
        if self.train_df is not None and self.val_df is not None:
            df = pd.concat([self.train_df, self.val_df])
        else:
            df = self._load_and_clean()
            
        total_samples = len(df)
        if total_samples == 0:
            return 0.0, 0

        over_threshold_count = 0
        max_len = 0
        lengths = []
        
        print(f"🔍 檢查超過 {threshold} tokens 的資料詳細資訊：")
        
        for idx, row in df.iterrows():
            # 重建與訓練時相同的格式化文本
            # 注意：必須與 KTDynamicDataset 使用相同格式
            # 新資料集已將對話整合為單一 Dialog 欄位
            formatted_text = (
                f"章節 : {row['chapter']}\n"
                f"知識點 : {row['section']}\n"
                f"學生掌握度 : [MASK]\n"
                f"簡答題作答紀錄 :\n{row['Short_Answer_Log']}\n"
            )
            
            # 計算長度 (不截斷)
            tokens = tokenizer.encode(formatted_text, add_special_tokens=True)
            seq_len = len(tokens)
            char_len = len(formatted_text)
            lengths.append(seq_len)
            
            if seq_len > max_len:
                max_len = seq_len
            
            if seq_len > threshold:
                over_threshold_count += 1
                # 嘗試取得 user_id，若無則顯示 Index
                uid = row.get('user_id', idx)
                print(f"  ⚠️ [過長] ID: {uid} | Token長度: {seq_len} | 文本長度: {char_len} | 章節: {row['chapter']} | 知識點: {row['section']}")
        
        ratio = (over_threshold_count / total_samples) * 100
        
        print(f"--- 長度分析報告 ---")
        print(f"總筆數: {total_samples}")
        print(f"最大長度: {max_len}")
        print(f"平均長度: {sum(lengths)/len(lengths):.1f}")
        print(f"超過 {threshold} 的筆數: {over_threshold_count}")
        print(f"超過比例: {ratio:.2f}%")
        
        return ratio, max_len


class KTDynamicDataset(Dataset):
    """
    功用：PyTorch Dataset 類別，負責在訓練時「動態」組合文本。
    它會在模型需要資料時 (即 __getitem__ 被呼叫時) 才即時組合字串並進行 tokenize。
    """
    def __init__(self, dataframe: pd.DataFrame, tokenizer: BertTokenizer, max_token_len: int = 512):
        """
        Args:
            dataframe (pd.DataFrame): 包含所需欄位的 DataFrame (例如 train_df 或 val_df)
            tokenizer (BertTokenizer): Hugging Face 的 Tokenizer
            max_token_len (int): 最大的序列長度
        """
        self.tokenizer = tokenizer
        self.data = dataframe.reset_index(drop=True)
        self.max_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        """
        當 DataLoader 請求一筆資料時，此方法會被觸發。
        """
        # 1. 根據索引取得單筆資料
        row = self.data.iloc[index]

        # 2. 提取所有需要的欄位 (在這裡處理，確保都是字串)
        # 新資料集使用 Short_Answer_Log 欄位
        chapter = str(row['chapter'])
        section = str(row['section'])
        short_answer_log = str(row['Short_Answer_Log'])
        label = int(row['labels'])

        # 3. 【核心：動態文本合併】
        # 根據新資料集結構組合字串
        # 注意：學生掌握度使用 [MASK] 隱藏，避免資料洩漏
        formatted_text = (
            f"章節 : {chapter}\n"
            f"知識點 : {section}\n"
            f"學生掌握度 : [MASK]\n"
            f"簡答題作答紀錄 :\n{short_answer_log}\n"
        )

        # print(f"formatted_text: {formatted_text}, length: {len(formatted_text)}")

        # 4. 將組合好的文本 Tokenize
        encoding = self.tokenizer(
            formatted_text,
            add_special_tokens=True, # Tokenizer 會自動在開頭加上 [CLS] 並在結尾加上 [SEP]
            max_length=self.max_len,
            padding="max_length",    # 填充到 max_len
            truncation=True,         # 截斷超過 max_len 的部分
            return_tensors="pt",     # 返回 PyTorch tensors
        )

        # 5. 回傳模型需要的字典 (並移除多餘的維度)
        return {
            'input_ids': encoding['input_ids'][0],
            'attention_mask': encoding['attention_mask'][0],
            'labels': torch.tensor(label, dtype=torch.long)
        }


# ========================================
# 自訂 Trainer：支援類別加權 (Class Weights)
# ========================================
class WeightedTrainer(Trainer):
    """
    自訂 Trainer，支援類別加權以處理類別不平衡問題。
    """
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        覆寫 compute_loss 方法，使用加權的 CrossEntropyLoss
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.class_weights is not None:
            loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fn = nn.CrossEntropyLoss()
        
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


class KTFinetuner:
    """
    功用：通用的知識追蹤模型訓練類別。
    支援 BERT、RoBERTa 等多種 Transformer 模型。
    """
    def __init__(self, model_name: str, data_processor: KTDataProcessor, 
                 training_args: TrainingArguments, max_token_len: int = 512,
                 class_weights: torch.Tensor = None, early_stopping_patience: int = None):
        """
        Args:
            model_name (str): 模型名稱，必須在 MODEL_CONFIGS 中定義
            data_processor (KTDataProcessor): 已經準備好資料的 DataProcessor 物件
            training_args (TrainingArguments): Hugging Face 的訓練參數
            max_token_len (int): 最大 Token 長度，預設 512
            class_weights (torch.Tensor): 類別權重，用於處理類別不平衡
            early_stopping_patience (int): Early Stopping 耐心値，幾個 epoch 無進步則停止
        """
        self.model_name = model_name
        self.processor = data_processor
        self.training_args = training_args
        self.max_token_len = max_token_len
        self.class_weights = class_weights
        self.early_stopping_patience = early_stopping_patience
        
        # 檢查模型是否在配置字典中
        if model_name not in MODEL_CONFIGS:
            raise ValueError(
                f"模型 '{model_name}' 不在支援列表中。\n"
                f"支援的模型: {list(MODEL_CONFIGS.keys())}"
            )
        
        model_config = MODEL_CONFIGS[model_name]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"模型將在 {self.device} 上運行")
        print(f"使用模型: {model_config['description']}")

        # 1. 載入 Tokenizer（使用配置字典中的類別）
        tokenizer_class = model_config["tokenizer_class"]
        self.tokenizer = tokenizer_class.from_pretrained(self.model_name)

        # 2. 載入模型（使用配置字典中的類別）
        config_class = model_config["config_class"]
        model_class = model_config["model_class"]
        
        config = config_class.from_pretrained(
            self.model_name,
            num_labels=self.processor.num_labels,
            id2label=self.processor.id2label,
            label2id=self.processor.label_map
        )
        self.model = model_class.from_pretrained(
            self.model_name,
            config=config
        ).to(self.device)
        
        self.trainer = None

    @staticmethod
    def _compute_metrics(eval_pred):
        """
        [靜態方法] 用於計算評估指標
        包括整體 accuracy 和每個類別的 precision, recall, f1-score, accuracy
        """
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        # 確保 labels 和 predictions 是 numpy arrays
        labels = np.array(labels)
        predictions = np.array(predictions)
        
        # 整體 accuracy
        overall_acc = accuracy_score(labels, predictions)
        
        # 每個類別的 precision, recall, f1-score
        # average=None 會返回每個類別的指標
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, labels=[0, 1, 2], average=None, zero_division=0
        )
        
        # 定義類別名稱（對應 label 0, 1, 2）- 3 個等級
        label_names = ["待加強", "尚可", "精熟"]
        
        # 計算 macro-F1（更適合不平衡的三分類問題）
        macro_f1 = f1.mean()
        
        # 建立結果字典
        metrics = {
            "accuracy": overall_acc,
            "macro_f1": macro_f1  # 用於 Optuna 優化
        }
        
        # 為每個類別計算指標
        for idx, label_name in enumerate(label_names):
            metrics[f"{label_name}_precision"] = precision[idx]
            metrics[f"{label_name}_recall"] = recall[idx]
            metrics[f"{label_name}_f1"] = f1[idx]
            
            # 計算每個類別的 accuracy
            # accuracy = 該類別預測正確的數量 / 該類別的總數量
            class_mask = labels == idx
            if class_mask.sum() > 0:
                class_acc = (predictions[class_mask] == labels[class_mask]).sum() / class_mask.sum()
                metrics[f"{label_name}_accuracy"] = float(class_acc)
            else:
                metrics[f"{label_name}_accuracy"] = 0.0
        
        return metrics

    def run_finetuning(self):
        """
        執行 Finetune 的主要流程
        """
        # 1. 從 processor 獲取資料
        train_df, val_df = self.processor.get_dataframes()
        
        # 2. 建立 Dataset 物件
        train_dataset = KTDynamicDataset(train_df, self.tokenizer, max_token_len=self.max_token_len)
        val_dataset = KTDynamicDataset(val_df, self.tokenizer, max_token_len=self.max_token_len)

        # 3. 設定 callbacks
        callbacks = []
        if self.early_stopping_patience is not None:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=self.early_stopping_patience))
            print(f"🚨 Early Stopping 已啟用，耐心値: {self.early_stopping_patience} epochs")

        # 4. 初始化 Trainer (使用 WeightedTrainer 如果有類別權重)
        if self.class_weights is not None:
            self.trainer = WeightedTrainer(
                class_weights=self.class_weights,
                model=self.model,
                args=self.training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=self._compute_metrics,
                callbacks=callbacks if callbacks else None,
            )
            print("⚖️ 使用加權 Trainer (WeightedTrainer)")
        else:
            self.trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=self._compute_metrics,
                callbacks=callbacks if callbacks else None,
            )

        # 4. 開始訓練
        print("--- 開始 Finetune ---")
        train_output = self.trainer.train()
        print("--- 訓練完成 ---")

        # 5. 評估最佳模型
        print("--- 評估最佳模型 ---")
        eval_results = self.trainer.evaluate()
        print(eval_results)
        
        # 6. 生成混淆矩陣
        print("\n--- 生成混淆矩陣 ---")
        val_predictions = self.trainer.predict(val_dataset)
        y_pred = np.argmax(val_predictions.predictions, axis=1)
        y_true = val_predictions.label_ids
        
        # 使用 TrainingVisualizer 繪製混淆矩陣
        # 從 MODEL_CONFIGS 取得模型描述作為圖表標題
        model_description = MODEL_CONFIGS.get(self.model_name, {}).get("description", self.model_name)
        visualizer = TrainingVisualizer(self.training_args.output_dir, model_name=model_description)
        class_names = [self.processor.id2label[i] for i in range(self.processor.num_labels)]
        visualizer.plot_confusion_matrix(y_true, y_pred, class_names)

        # 7. 視覺化訓練結果 (重用上面的 visualizer)
        if self.trainer.state.log_history:
            visualizer.plot(self.trainer.state.log_history)
        else:
            print("⚠️ 無訓練紀錄，跳過繪圖。")

    def save_model(self, save_path: str):
        """
        儲存訓練好的模型和 Tokenizer
        
        Args:
            save_path (str): 儲存的路徑
        """
        if self.trainer is None:
            raise RuntimeError("模型尚未訓練，請先呼叫 .run_finetuning()")
            
        self.trainer.save_model(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"模型與 Tokenizer 已儲存至: {save_path}")
    
    def run_finetuning_for_optuna(self):
        """
        簡化版訓練流程（用於 Optuna 超參數搜索）
        不生成視覺化圖表以加快速度
        """
        # 資料已在 objective 函數中準備好，這裡直接取用
        train_df, val_df = self.processor.get_dataframes()
        
        # 2. 建立 Dataset
        train_dataset = KTDynamicDataset(train_df, self.tokenizer, max_token_len=self.max_token_len)
        val_dataset = KTDynamicDataset(val_df, self.tokenizer, max_token_len=self.max_token_len)
        
        # 3. 初始化 Trainer
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics,
        )
        
        # 4. 訓練
        self.trainer.train()
        
        # 5. 評估
        eval_results = self.trainer.evaluate()
        
        return eval_results


class HyperparameterSearcher:
    """
    Optuna 超參數搜索器
    
    通用的超參數搜索類，可用於不同的模型類型（BERT, RoBERTa, Longformer等）
    """
    
    def __init__(self, model_class, data_processor_class, search_space: dict = None):
        """
        Args:
            model_class: 模型訓練類（如 BertKTFinetuner）
            data_processor_class: 資料處理類（如 KTDataProcessor）
            search_space: 超參數搜索空間配置
        """
        self.model_class = model_class
        self.data_processor_class = data_processor_class
        
        # 默認搜索空間
        self.search_space = search_space or {
            "learning_rate": {"low": 1e-5, "high": 1e-4, "log": True},
            "num_train_epochs": {"values": [5, 10, 15]},
            "per_device_train_batch_size": {"values": [4, 8, 16]},
            "warmup_steps": {"values": [50, 100, 150, 200]},
            "weight_decay": {"low": 0.0, "high": 0.1}
        }
    
    def run_search(
        self,
        csv_path: str,
        model_name: str = "bert-base-chinese",
        output_base_dir: str = "./optuna_results",
        n_trials: int = 15,
        study_name: str = "hp_search"
    ):
        # 執行超參數搜索
        print("開始 Optuna 超參數搜索\n")
        print(f"將嘗試 {n_trials} 組參數組合\n")
        
        os.makedirs(output_base_dir, exist_ok=True)
        
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5)
        )
        
        def objective(trial):
            params = self._suggest_parameters(trial)
            trial_output_dir = os.path.join(output_base_dir, f"trial_{trial.number}")
            os.makedirs(trial_output_dir, exist_ok=True)
            
            training_args_dict = {
                "output_dir": trial_output_dir,
                "num_train_epochs": params["num_train_epochs"],
                "per_device_train_batch_size": params["per_device_train_batch_size"],
                "per_device_eval_batch_size": 4,
                "learning_rate": params["learning_rate"],
                "warmup_steps": params["warmup_steps"],
                "weight_decay": params["weight_decay"],
                "logging_dir": f"{trial_output_dir}/logs",
                "logging_steps": 10,
                "eval_strategy": "epoch",
                "save_strategy": "no",  # 不保存 checkpoint，節省空間
                "metric_for_best_model": "macro_f1",
                "report_to": "none",
                "seed": 42,
                "data_seed": 42
            }
            
            try:
                self._print_trial_info(trial.number, params)
                
                data_processor = self.data_processor_class(csv_path=csv_path)
                data_processor.prepare_data(test_size=0.2, random_state=42)
                
                training_args = TrainingArguments(**training_args_dict)
                
                finetuner = self.model_class(
                    model_name=model_name,
                    data_processor=data_processor,
                    training_args=training_args,
                    max_token_len=512
                )
                
                eval_results = finetuner.run_finetuning_for_optuna()
                score = eval_results.get("eval_macro_f1", eval_results["eval_accuracy"])
                
                print(f"Trial {trial.number} 完成: macro_f1 = {score:.4f}")
                
                trial.report(score, step=params["num_train_epochs"])
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
                return score
                
            except Exception as e:
                print(f"❌ Trial {trial.number} 失敗: {str(e)}")
                raise
        
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True, catch=(Exception,))
        self._print_results(study, output_base_dir)
        
        return study
    
    def _suggest_parameters(self, trial):
        params = {}
        for param_name, config in self.search_space.items():
            if "values" in config:
                params[param_name] = trial.suggest_categorical(param_name, config["values"])
            elif "low" in config and "high" in config:
                if config.get("log", False):
                    params[param_name] = trial.suggest_float(
                        param_name, config["low"], config["high"], log=True
                    )
                else:
                    params[param_name] = trial.suggest_float(
                        param_name, config["low"], config["high"]
                    )
        return params
    
    def _print_trial_info(self, trial_number, params):
        print(f"{'='*80}")
        print(f"Trial {trial_number}: 測試參數組合")
        print(f"  Learning Rate: {params['learning_rate']:.2e}")
        print(f"  Epochs: {params['num_train_epochs']}")
        print(f"  Batch Size: {params['per_device_train_batch_size']}")
        print(f"  Warmup Steps: {params['warmup_steps']}")
        print(f"  Weight Decay: {params['weight_decay']:.4f}")
        print(f"{'='*80}")
    
    def _print_results(self, study, output_base_dir):
        print(f"{'='*80}")
        print(f"🎉 超參數搜索完成!")
        print(f"{'='*80}")
        
        print(f"📊 最佳參數:")
        for key, value in study.best_params.items():
            if 'learning_rate' in key:
                print(f"  {key}: {value:.2e}")
            else:
                print(f"  {key}: {value}")
        
        print(f"✅ 最佳驗證 macro-F1: {study.best_value:.4f}")
        print(f"✅ 最佳 Trial 編號: {study.best_trial.number}")
        
        study_results_path = os.path.join(output_base_dir, "study_results.csv")
        df = study.trials_dataframe()
        df.to_csv(study_results_path, index=False)
        print(f"💾 Study 結果已儲存至: {study_results_path}")
        
        try:
            print(f"📈 正在生成可視化圖表...")
            fig1 = plot_optimization_history(study)
            fig1_path = os.path.join(output_base_dir, "optimization_history.png")
            fig1.write_image(fig1_path)
            print(f"  ✅ 優化歷程圖: {fig1_path}")
            
            if len(study.trials) >= 3:
                fig2 = plot_param_importances(study)
                fig2_path = os.path.join(output_base_dir, "param_importances.png")
                fig2.write_image(fig2_path)
                print(f"  ✅ 參數重要性圖: {fig2_path}")
        except Exception as e:
            print(f"⚠️ 可視化生成失敗: {e}")


def ensure_dir_exists(path: str):
    """確保目標資料夾及其所有父資料夾存在"""
    os.makedirs(path, exist_ok=True)
    print(f"✓ 資料夾已確認存在: {path}")



def generate_comparison_report(bert_results: dict, roberta_results: dict, output_path: str):
    """
    生成 BERT vs RoBERTa 比較報告並保存為 CSV
    
    Args:
        bert_results: BERT 訓練結果字典
        roberta_results: RoBERTa 訓練結果字典
        output_path: CSV 保存路徑
    """
    import pandas as pd
    from datetime import datetime
    
    print(f"\n{'='*80}")
    print("📊 生成模型比較報告")
    print(f"{'='*80}\n")
    
    # 準備比較數據
    comparison_data = {
        "模型名稱": ["BERT (bert-base-chinese)", "RoBERTa (hfl/chinese-roberta-wwm-ext)"],
        "Macro F1-Score": [
            bert_results.get("eval_macro_f1", 0),
            roberta_results.get("eval_macro_f1", 0)
        ],
        "整體準確率": [
            bert_results.get("eval_accuracy", 0),
            roberta_results.get("eval_accuracy", 0)
        ],
        "待加強_F1": [
            bert_results.get("eval_待加強_f1", 0),
            roberta_results.get("eval_待加強_f1", 0)
        ],
        "尚可_F1": [
            bert_results.get("eval_尚可_f1", 0),
            roberta_results.get("eval_尚可_f1", 0)
        ],
        "精熟_F1": [
            bert_results.get("eval_精熟_f1", 0),
            roberta_results.get("eval_精熟_f1", 0)
        ],
        "模型路徑": [
            bert_results.get("model_path", "N/A"),
            roberta_results.get("model_path", "N/A")
        ],
        "訓練時間戳": [
            bert_results.get("timestamp", "N/A"),
            roberta_results.get("timestamp", "N/A")
        ]
    }
    
    df = pd.DataFrame(comparison_data)
    
    # 保存 CSV
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ 比較報告已保存至: {output_path}\n")
    
    # 打印到終端
    print("="*80)
    print("📊 BERT vs RoBERTa 性能比較")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    # 判斷最佳模型
    bert_f1 = bert_results.get("eval_macro_f1", 0)
    roberta_f1 = roberta_results.get("eval_macro_f1", 0)
    
    if roberta_f1 > bert_f1:
        improvement = ((roberta_f1 - bert_f1) / bert_f1) * 100
        print(f"\n🏆 最佳模型: RoBERTa")
        print(f"📈 性能提升: {improvement:.2f}%")
        print(f"   Macro-F1: {bert_f1:.4f} → {roberta_f1:.4f}")
    elif bert_f1 > roberta_f1:
        improvement = ((bert_f1 - roberta_f1) / roberta_f1) * 100
        print(f"\n🏆 最佳模型: BERT")
        print(f"📈 性能優勢: {improvement:.2f}%")
        print(f"   Macro-F1: {roberta_f1:.4f} → {bert_f1:.4f}")
    else:
        print(f"\n⚖️  兩個模型性能相當")
        print(f"   Macro-F1: {bert_f1:.4f}")
    
    print("="*80 + "\n")
    
    return df


if __name__ == "__main__":
    
    # ========================================
    # 🎯 雙模型比較實驗配置
    # ========================================
    
    print("\n" + "="*80)
    print("🔬 BERT vs RoBERTa 知識追蹤模型比較實驗")
    print("="*80 + "\n")
    
    # 共用配置
    DATASET_PATH = "datasets/finetune_dataset_1142_v4_without_chat_0227.csv"
    N_TRIALS = 15  # Optuna 搜索次數
    FULL_TRAIN_EPOCHS = 15  # 減少訓練輪數，配合 Early Stopping
    
    # 記錄訓練結果
    training_results = {}
    
    # ========================================
    # 實驗 1: BERT 模型 (跳過)
    # ========================================
    """
    print("\n" + "="*80)
    print("🔍 實驗 1/2: BERT 模型 (bert-base-chinese)")
    print("="*80 + "\n")
    
    bert_model_name = "bert-base-chinese"
    bert_output_base = "./optuna_results/bert"
    
    # BERT Optuna 搜索
    print(f"開始 BERT 超參數搜索 ({N_TRIALS} trials)...\n")
    
    bert_searcher = HyperparameterSearcher(
        model_class=KTFinetuner,
        data_processor_class=KTDataProcessor
    )
    
    bert_study = bert_searcher.run_search(
        csv_path=DATASET_PATH,
        model_name=bert_model_name,
        output_base_dir=bert_output_base,
        n_trials=N_TRIALS,
        study_name="bert_hp_search"
    )
    
    # BERT 完整訓練
    print("\n" + "="*80)
    print(f"🚀 使用最佳參數訓練 BERT ({FULL_TRAIN_EPOCHS} epochs)")
    print("="*80 + "\n")
    
    bert_best_params = bert_study.best_params
    timestamp_bert = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 符合使用者要求的命名格式
    bert_output_dir = f"./results/bert-base-chinese_{timestamp_bert}"
    bert_model_path = f"{bert_output_dir}/final_model"
    
    ensure_dir_exists(bert_output_dir)
    ensure_dir_exists(bert_model_path)
    
    bert_training_args = TrainingArguments(
        output_dir=bert_output_dir,
        num_train_epochs=FULL_TRAIN_EPOCHS,
        per_device_train_batch_size=bert_best_params.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=4,
        learning_rate=bert_best_params.get("learning_rate", 3e-5),
        warmup_steps=bert_best_params.get("warmup_steps", 100),
        weight_decay=bert_best_params.get("weight_decay", 0.01),
        logging_dir=f"{bert_output_dir}/logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,  # 只保留最近 2 個 checkpoint，節省空間
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        report_to="none",
        seed=42,
        data_seed=42
    )
    
    bert_data_processor = KTDataProcessor(csv_path=DATASET_PATH)
    bert_data_processor.prepare_data(test_size=0.2, random_state=42)
    
    bert_finetuner = KTFinetuner(
        model_name=bert_model_name,
        data_processor=bert_data_processor,
        training_args=bert_training_args,
        max_token_len=512
    )
    
    bert_finetuner.run_finetuning()
    bert_finetuner.save_model(save_path=bert_model_path)
    
    # 獲取 BERT 最終評估結果
    bert_eval_results = bert_finetuner.trainer.evaluate()
    training_results["bert"] = {
        **bert_eval_results,
        "model_path": bert_model_path,
        "timestamp": timestamp_bert
    }
    
    print(f"\n✅ BERT 訓練完成！模型保存至: {bert_model_path}\n")
    """
    print("⚠️ Skipping BERT training as requested. Only training RoBERTa.")
    training_results["bert"] = None # Placeholder
    
    # ========================================
    # 實驗 2: RoBERTa 模型
    # ========================================
    
    print("\n" + "="*80)
    print("🔍 實驗 2/2: RoBERTa 模型 (hfl/chinese-roberta-wwm-ext)")
    print("="*80 + "\n")
    
    roberta_model_name = "hfl/chinese-roberta-wwm-ext"
    roberta_output_base = "./optuna_results/roberta"
    
    # RoBERTa Optuna 搜索
    print(f"開始 RoBERTa 超參數搜索 ({N_TRIALS} trials)...\n")
    
    roberta_searcher = HyperparameterSearcher(
        model_class=KTFinetuner,
        data_processor_class=KTDataProcessor
    )
    
    roberta_study = roberta_searcher.run_search(
        csv_path=DATASET_PATH,
        model_name=roberta_model_name,
        output_base_dir=roberta_output_base,
        n_trials=N_TRIALS,
        study_name="roberta_hp_search"
    )
    
    # RoBERTa 完整訓練
    print("\n" + "="*80)
    print(f"🚀 使用最佳參數訓練 RoBERTa ({FULL_TRAIN_EPOCHS} epochs)")
    print("="*80 + "\n")
    
    roberta_best_params = roberta_study.best_params
    timestamp_roberta = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 符合使用者要求的命名格式
    roberta_output_dir = f"./results/roberta-chinese_{timestamp_roberta}"
    roberta_model_path = f"{roberta_output_dir}/final_model"
    
    ensure_dir_exists(roberta_output_dir)
    ensure_dir_exists(roberta_model_path)
    
    roberta_training_args = TrainingArguments(
        output_dir=roberta_output_dir,
        num_train_epochs=FULL_TRAIN_EPOCHS,
        per_device_train_batch_size=roberta_best_params.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=4,
        learning_rate=roberta_best_params.get("learning_rate", 3e-5) * 0.5,  # 調低 Learning Rate
        warmup_steps=roberta_best_params.get("warmup_steps", 100) + 100,  # 增加 Warmup Steps
        weight_decay=roberta_best_params.get("weight_decay", 0.01),
        logging_dir=f"{roberta_output_dir}/logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,  # 只保留最近 2 個 checkpoint，節省空間
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,  # macro_f1 越高越好
        report_to="none",
        seed=42,
        data_seed=42
    )
    
    roberta_data_processor = KTDataProcessor(csv_path=DATASET_PATH)
    roberta_data_processor.prepare_data(test_size=0.2, random_state=42)
    
    # 計算類別權重 (基於訓練集分佈)
    train_df, _ = roberta_data_processor.get_dataframes()
    label_counts = train_df['labels'].value_counts().sort_index()
    total_samples = len(train_df)
    # 權重 = 總樣本數 / (類別數 * 該類別樣本數)
    class_weights = torch.tensor([
        total_samples / (3 * label_counts[0]),  # 待加強
        total_samples / (3 * label_counts[1]),  # 尚可
        total_samples / (3 * label_counts[2])   # 精熟
    ], dtype=torch.float32)
    print(f"\n📋 類別權重: 待加強={class_weights[0]:.2f}, 尚可={class_weights[1]:.2f}, 精熟={class_weights[2]:.2f}")
    
    roberta_finetuner = KTFinetuner(
        model_name=roberta_model_name,
        data_processor=roberta_data_processor,
        training_args=roberta_training_args,
        max_token_len=512,
        class_weights=class_weights,  # 傳入類別權重
        early_stopping_patience=3     # Early Stopping: 3 個 epoch 無進步則停止
    )
    
    roberta_finetuner.run_finetuning()
    roberta_finetuner.save_model(save_path=roberta_model_path)
    
    # 獲取 RoBERTa 最終評估結果
    roberta_eval_results = roberta_finetuner.trainer.evaluate()
    training_results["roberta"] = {
        **roberta_eval_results,
        "model_path": roberta_model_path,
        "timestamp": timestamp_roberta
    }
    
    print(f"\n✅ RoBERTa 訓練完成！模型保存至: {roberta_model_path}\n")

    # ========================================
    # 📊 生成比較報告 (if BERT results exist)
    # ========================================
    
    if training_results.get("bert") is not None:
        comparison_output = "./results/model_comparison_report.csv"
        ensure_dir_exists("./results")
        
        generate_comparison_report(
            bert_results=training_results["bert"],
            roberta_results=training_results["roberta"],
            output_path=comparison_output
        )
        
        print("\n" + "="*80)
        print("🎉 雙模型比較實驗完成！")
        print("="*80)
        print(f"📁 BERT 模型: {training_results['bert']['model_path']}")
        print(f"📁 RoBERTa 模型: {roberta_model_path}")
        print(f"📊 比較報告: {comparison_output}")
        print("="*80 + "\n")
    else:
        print("\n" + "="*80)
        print("🎉 RoBERTa 訓練實驗完成！(BERT 訓練已跳過)")
        print("="*80)
        print(f"📁 RoBERTa 模型: {roberta_model_path}")
        print("="*80 + "\n")
