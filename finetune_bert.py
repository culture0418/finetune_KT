import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from datetime import datetime
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import (
    BertConfig,
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer
)

class TrainingVisualizer:
    """
    功用：負責將訓練過程中的 Log 轉化為圖表與 CSV 報表。
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        ensure_dir_exists(self.output_dir)
        # 設定非互動式後端，避免在 Server 上報錯
        matplotlib.use('Agg')

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

        # 2. 繪製圖表
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('BERT Knowledge Tracing - Training History', fontsize=16, fontweight='bold')

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
        
        # 3. 儲存檔案
        # 使用更具描述性的檔名
        img_path = os.path.join(self.output_dir, 'training_metrics_visualization.png')
        plt.savefig(img_path, dpi=300)
        print(f"✅ 圖表已儲存至: {img_path}")
        plt.close(fig) # 釋放記憶體

        # 4. 儲存 CSV 摘要
        if eval_accs:
            df = pd.DataFrame({
                'Epoch': epochs,
                'Eval_Loss': eval_losses,
                'Eval_Accuracy': eval_accs
            })
            csv_path = os.path.join(self.output_dir, 'training_metrics_summary.csv')
            df.to_csv(csv_path, index=False)
            print(f"✅ 摘要已儲存至: {csv_path}")


class KTDataProcessor:
    """
    功用：專門處理 finetune_dataset_k4_global.csv 檔案。
    負責載入、清理、轉換標籤，並分割資料集。
    """
    def __init__(self, csv_path: str):
        """
        初始化物件，傳入 CSV 檔案的路徑。
        
        Args:
            csv_path (str): 'finetune_dataset.csv' 的路徑
        """
        self.csv_path = csv_path
        # 定義欄位和標籤
        self.required_cols = ['chapter', 'section', 'all_logs', 'Preview_ChatLog', 'Review_ChatLog', 'Mastery_Level_K4']
        self.label_map = {"待加強": 0, "尚可": 1, "良好": 2, "精熟": 3}
        self.id2label = {v: k for k, v in self.label_map.items()}
        self.num_labels = len(self.label_map)
        
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
            df = pd.read_csv(self.csv_path)
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
        text_cols = [c for c in self.required_cols if c != 'Mastery_Level_K4']
        for col in text_cols:
            # fillna('') 把 NaN 變成空字串
            # astype(str) 確保即使 CSV 裡是數字 123，也會變成字串 "123"
            df[col] = df[col].fillna('').astype(str)

        # 3. 清理標籤欄位 (Mastery_Level_K4)
        # 步驟 A: 先把 Mastery_Level_K4 本身是 NaN/空值的去掉
        if df['Mastery_Level_K4'].isnull().any():
            n_missing = df['Mastery_Level_K4'].isnull().sum()
            print(f"警告：移除 {n_missing} 筆 'Mastery_Level_K4' 為空的資料")
            df = df.dropna(subset=['Mastery_Level_K4'])

        # 步驟 B: 進行 Mapping
        df['labels'] = df['Mastery_Level_K4'].map(self.label_map)

        # 步驟 C: 檢查 Mapping 後是否產生 NaN (代表出現了字典裡沒有的標籤，如 'Unknown')
        if df['labels'].isnull().any():
            invalid_rows = df[df['labels'].isnull()]
            invalid_values = invalid_rows['Mastery_Level_K4'].unique()
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
        使用 stratify 確保訓練集和驗證集中，四種標籤（待加強、尚可、良好、精熟）的比例相同。
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
            # 重現 KTDynamicDataset 的組字邏輯 (必須保持一致)
            formatted_text = (
                f"章節 : {row['chapter']}\n"
                f"知識點 : {row['section']}\n"
                f"學生掌握度 : {row['Mastery_Level_K4']} [MASK]\n"
                f"作答紀錄 :\n{row['all_logs']}\n"
                f"[課前相關對話紀錄]\n{row['Preview_ChatLog']}\n"
                f"[課後相關對話紀錄]\n{row['Review_ChatLog']}\n"
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
        chapter = str(row['chapter'])
        section = str(row['section'])
        all_logs = str(row['all_logs'])
        preview_chat = str(row['Preview_ChatLog'])
        review_chat = str(row['Review_ChatLog'])
        mastery_text = str(row['Mastery_Level_K4'])
        label = int(row['labels'])

        # 3. 【核心：動態文本合併】
        # 根據您的範本即時組合字串
        formatted_text = (
            f"章節 : {chapter}\n"
            f"知識點 : {section}\n"
            f"學生掌握度 : {mastery_text} [MASK]\n"
            f"作答紀錄 :\n{all_logs}\n"
            f"[課前相關對話紀錄]\n{preview_chat}\n"
            f"[課後相關對話紀錄]\n{review_chat}\n"
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
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BertKTFinetuner:
    """
    功用：主要的協同運作類別。
    負責初始化模型、Tokenizer、Trainer，並執行訓練和評估。
    """
    def __init__(self, model_name: str, data_processor: KTDataProcessor, training_args: TrainingArguments):
        """
        Args:
            model_name (str): 要從 Hugging Face 載入的模型名稱
            data_processor (KTDataProcessor): 已經準備好資料的 DataProcessor 物件
            training_args (TrainingArguments): Hugging Face 的訓練參數
        """
        self.model_name = model_name
        self.processor = data_processor
        self.training_args = training_args
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"模型將在 {self.device} 上運行")

        # 1. 載入 Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)

        # 2. 載入模型 (使用 BertConfig 來傳遞標籤資訊)
        config = BertConfig.from_pretrained(
            self.model_name,
            num_labels=self.processor.num_labels,
            id2label=self.processor.id2label,
            label2id=self.processor.label_map
        )
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            config=config
        ).to(self.device)
        
        self.trainer = None # Trainer 會在 run_finetuning 時被初始化

    @staticmethod
    def _compute_metrics(eval_pred):
        """
        [靜態方法] 用於計算評估指標 (準確率)
        """
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions)
        }

    def run_finetuning(self):
        """
        執行 Finetune 的主要流程
        """
        # 1. 從 processor 獲取資料
        train_df, val_df = self.processor.get_dataframes()
        
        # 2. 建立 Dataset 物件
        train_dataset = KTDynamicDataset(train_df, self.tokenizer)
        val_dataset = KTDynamicDataset(val_df, self.tokenizer)

        # 3. 初始化 Trainer
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics,
        )

        # 4. 開始訓練
        print("--- 開始 Finetune ---")
        train_output = self.trainer.train()
        print("--- 訓練完成 ---")

        # 5. 評估最佳模型
        print("--- 評估最佳模型 ---")
        eval_results = self.trainer.evaluate()
        print(eval_results)

        # 6. 視覺化訓練結果 (使用新 Class)
        if self.trainer.state.log_history:
            visualizer = TrainingVisualizer(self.training_args.output_dir)
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


def ensure_dir_exists(path: str):
    """
    確保目標資料夾及其所有父資料夾存在。
    如果不存在，則遞迴創建。
    
    Args:
        path (str): 目標資料夾路徑
    """
    os.makedirs(path, exist_ok=True)
    print(f"✓ 資料夾已確認存在: {path}")

# --- 主程式執行區塊 ---
if __name__ == "__main__":
    
    # --- 1. 定義所有設定 ---
    
    # 檔案和模型路徑
    CONFIG = {
        "csv_path": "datasets/finetune_dataset_k4_global.csv",
        "model_name": "bert-base-chinese",
        "output_base_dir": "./results",
        "final_model_dir": "final_model",
        "test_size": 0.2,
        "max_token_len": 512,
    }

    # 生成時間戳記（格式：YYYYMMDD_HHMMSS）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 動態生成訓練結果資料夾路徑
    OUTPUT_DIR = f"{CONFIG['output_base_dir']}/{CONFIG['model_name']}_{timestamp}"
    FINAL_MODEL_PATH = f"{OUTPUT_DIR}/{CONFIG['final_model_dir']}"
    
    # 確保資料夾存在
    ensure_dir_exists(OUTPUT_DIR)
    ensure_dir_exists(FINAL_MODEL_PATH)
    
    print(f"\n📁 訓練結果將儲存至: {OUTPUT_DIR}")
    print(f"📁 最終模型將儲存至: {FINAL_MODEL_PATH}\n")
    
    # 訓練參數 (打包成一個字典)
    # 可以在這裡調整訓練的超參數
    training_args_dict = {
        "output_dir": OUTPUT_DIR,
        "num_train_epochs": 3,              # 訓練輪數
        "per_device_train_batch_size": 4,   # 每個 GPU/CPU 的 batch size
        "per_device_eval_batch_size": 4,    # 驗證時的 batch size
        "learning_rate": 5e-5,              # 學習率
        "warmup_steps": 50,                 # 預熱步數
        "weight_decay": 0.01,               # 權重衰減
        "logging_dir": f"{OUTPUT_DIR}/logs",# TensorBoard 日誌目錄
        "logging_steps": 10,                # 每幾步紀錄一次 log
        "eval_strategy": "epoch",           # 每個 epoch 結束後進行評估
        "save_strategy": "epoch",           # 每個 epoch 結束後儲存 checkpoint
        "load_best_model_at_end": True,     # 訓練結束後載入表現最好的模型
        "metric_for_best_model": "accuracy",# 以 accuracy 作為評估標準
        "report_to": "none"                 # 不上傳到 WandB 等平台
    }
    
    # --- 2. 執行資料處理 ---
    
    # 初始化資料處理器
    # 初始化資料處理器
    data_processor = KTDataProcessor(csv_path=CONFIG['csv_path'])
    # 執行資料準備 (載入、清理、分割)
    data_processor.prepare_data(test_size=CONFIG['test_size'])
    
    # 新增：執行 Token 長度分析
    ratio, max_len = data_processor.analyze_token_lengths(CONFIG['model_name'], threshold=CONFIG['max_token_len'])
    
    if ratio > 5.0:
        print(f"\n⚠️ 警告：有 {ratio:.2f}% 的資料長度超過 512 tokens (最大 {max_len})")
        print("建議考慮使用支援長文本的模型 (如 Longformer, BigBird) 或調整資料截斷策略。")
    else:
        print(f"\n✅ 資料長度檢查通過 (超過 512 的比例為 {ratio:.2f}%)")

    # --- 3. 執行模型 Finetune ---
    
    # 建立 TrainingArguments 物件
    training_args = TrainingArguments(**training_args_dict)
    
    # 初始化 Finetuner
    finetuner = BertKTFinetuner(
        model_name=CONFIG['model_name'],
        data_processor=data_processor, # 傳入處理好的資料
        training_args=training_args
    )
    
    # 開始訓練
    finetuner.run_finetuning()
    
    # --- 4. 儲存最終模型 ---
    finetuner.save_model(save_path=FINAL_MODEL_PATH)
    
    print(f"\n" + "="*80)
    print(f"✅ 訓練完成！")
    print(f"📁 所有訓練結果已儲存至: {OUTPUT_DIR}")
    print(f"📁 最終模型已儲存至: {FINAL_MODEL_PATH}")
    print(f"🕒 訓練時間戳記: {timestamp}")
    print("="*80)