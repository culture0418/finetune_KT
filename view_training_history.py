import json
import os
import glob
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

matplotlib.use('Agg')  # 使用非互動式後端，避免顯示視窗

# ============================================================================
# 設定區塊
# ============================================================================

# 若要指定特定的結果目錄，請將 None 改為字串，例如: "./results/bert-base-chinese_20240101_120000"
TARGET_DIR = None 

# 預設搜尋的基礎目錄
BASE_RESULTS_DIR = './results'

# ============================================================================
# 主程式邏輯
# ============================================================================

def find_latest_result_dir(base_dir):
    """尋找 base_dir 底下最新的 timestamp 目錄"""
    if not os.path.exists(base_dir):
        return None
    
    # 取得所有子目錄
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not subdirs:
        return None
        
    # 根據修改時間排序 (最新的在最後)
    latest_subdir = max(subdirs, key=os.path.getmtime)
    return latest_subdir

def main():
    # 1. 決定要分析的目錄
    if TARGET_DIR:
        target_dir = TARGET_DIR
        print(f"📌 使用指定的結果目錄: {target_dir}")
    else:
        print(f"🔍 正在搜尋最新的結果目錄...")
        target_dir = find_latest_result_dir(BASE_RESULTS_DIR)
        if target_dir:
            print(f"📌 找到最新的結果目錄: {target_dir}")
        else:
            print(f"❌ 錯誤：在 {BASE_RESULTS_DIR} 找不到任何結果目錄")
            return

    if not os.path.exists(target_dir):
        print(f"❌ 錯誤：目錄不存在: {target_dir}")
        return

    # 2. 找到所有 checkpoint 目錄
    # 搜尋模式：target_dir/checkpoint-*
    checkpoint_pattern = os.path.join(target_dir, 'checkpoint-*')
    checkpoint_dirs = sorted(glob.glob(checkpoint_pattern))

    if not checkpoint_dirs:
        print(f"❌ 錯誤：在 {target_dir} 找不到任何 checkpoint 目錄")
        print("請確認該目錄是否包含 checkpoint-*/ 子目錄")
        return

    print(f"找到 {len(checkpoint_dirs)} 個 checkpoint\n")

    # 3. 讀取訓練歷史
    all_logs = []
    best_metrics = []

    for checkpoint_dir in checkpoint_dirs:
        trainer_state_path = os.path.join(checkpoint_dir, 'trainer_state.json')
        
        try:
            with open(trainer_state_path, 'r') as f:
                state = json.load(f)
                
            # 收集訓練日誌
            all_logs.extend(state.get('log_history', []))
            
            # 收集最佳指標
            if 'best_metric' in state:
                best_metrics.append({
                    'checkpoint': checkpoint_dir,
                    'best_metric': state['best_metric'],
                    'best_model_checkpoint': state.get('best_model_checkpoint', 'N/A')
                })
                
            print(f"✓ 已讀取: {checkpoint_dir}")
            
        except FileNotFoundError:
            print(f"⚠ 跳過: {checkpoint_dir} (找不到 trainer_state.json)")
        except json.JSONDecodeError:
            print(f"⚠ 跳過: {checkpoint_dir} (JSON 格式錯誤)")

    # 4. 去除重複的日誌
    unique_logs = {}
    for log in all_logs:
        key = (log.get('epoch', 0), log.get('step', 0))
        if key not in unique_logs:
            unique_logs[key] = log

    # 按 epoch 排序
    sorted_logs = sorted(unique_logs.values(), key=lambda x: (x.get('epoch', 0), x.get('step', 0)))

    # 5. 顯示摘要
    print("\n" + "="*80)
    print("=== 訓練摘要 ===")
    print("="*80)

    # 找出整體最佳 accuracy
    eval_accuracies = [(log['epoch'], log['eval_accuracy']) for log in sorted_logs if 'eval_accuracy' in log]
    if eval_accuracies:
        best_epoch, best_acc = max(eval_accuracies, key=lambda x: x[1])
        print(f"🏆 整體最佳 Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%) at Epoch {best_epoch}")
    else:
        print("尚無 Evaluation 紀錄")

    # 6. 繪製圖表
    print("\n📊 正在繪製訓練曲線...")
    
    epochs = []
    train_losses = []
    eval_losses = []
    eval_accs = []

    for log in sorted_logs:
        epoch = log.get('epoch', 0)
        if 'loss' in log:
            train_losses.append({'epoch': epoch, 'loss': log['loss']})
        if 'eval_loss' in log:
            epochs.append(epoch)
            eval_losses.append(log['eval_loss'])
            eval_accs.append(log['eval_accuracy'])

    if not epochs and not train_losses:
        print("⚠️ 無足夠數據繪圖")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Training History - {os.path.basename(target_dir)}', fontsize=16, fontweight='bold')

    # Validation Loss
    if eval_losses:
        ax1.plot(epochs, eval_losses, marker='o', label='Val Loss', color='red')
        ax1.set_title('Validation Loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

    # Validation Accuracy
    if eval_accs:
        ax2.plot(epochs, eval_accs, marker='o', label='Val Acc', color='green')
        ax2.set_title('Validation Accuracy')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 標記最佳點
        max_acc = max(eval_accs)
        max_idx = eval_accs.index(max_acc)
        ax2.annotate(f'Best: {max_acc:.4f}', (epochs[max_idx], max_acc), 
                     xytext=(0, 10), textcoords='offset points', ha='center', color='green', fontweight='bold')

    # Training Loss
    if train_losses:
        t_epochs = [x['epoch'] for x in train_losses]
        t_vals = [x['loss'] for x in train_losses]
        ax3.plot(t_epochs, t_vals, label='Train Loss', color='blue', alpha=0.6)
        ax3.set_title('Training Loss')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

    plt.tight_layout()
    
    # 儲存圖片到該結果目錄下
    output_img_path = os.path.join(target_dir, 'training_history_curves.png')
    plt.savefig(output_img_path, dpi=300)
    print(f"✅ 訓練曲線圖已儲存至: {output_img_path}")

    # 儲存 CSV
    if eval_accs:
        df = pd.DataFrame({'Epoch': epochs, 'Eval_Loss': eval_losses, 'Eval_Accuracy': eval_accs})
        output_csv_path = os.path.join(target_dir, 'training_summary.csv')
        df.to_csv(output_csv_path, index=False)
        print(f"✅ 訓練摘要已儲存至: {output_csv_path}")

if __name__ == "__main__":
    main()
