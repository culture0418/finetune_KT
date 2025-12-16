import pytest
import os
import pandas as pd
from finetune_bert import TrainingVisualizer

class TestTrainingVisualizer:
    
    @pytest.fixture
    def mock_log_history(self):
        """
        模擬 Hugging Face Trainer 的 log_history
        包含 2 個 epoch 的訓練與驗證數據
        """
        return [
            {'loss': 0.8, 'epoch': 0.1, 'step': 10},
            {'loss': 0.6, 'epoch': 0.5, 'step': 50},
            {'eval_loss': 0.5, 'eval_accuracy': 0.80, 'epoch': 1.0, 'step': 100},
            {'loss': 0.4, 'epoch': 1.5, 'step': 150},
            {'eval_loss': 0.3, 'eval_accuracy': 0.90, 'epoch': 2.0, 'step': 200},
        ]

    def test_plot_creates_files(self, tmp_path, mock_log_history):
        """
        測試 plot 方法是否能成功建立 png 和 csv 檔案
        使用 tmp_path 避免在真實目錄產生垃圾檔案
        """
        # 1. 初始化 Visualizer (使用臨時目錄)
        output_dir = tmp_path / "viz_output"
        visualizer = TrainingVisualizer(str(output_dir))
        
        # 2. 執行繪圖
        visualizer.plot(mock_log_history)
        
        # 3. 驗證檔案是否存在 (使用更新後的檔名)
        expected_png = output_dir / "training_metrics_visualization.png"
        expected_csv = output_dir / "training_metrics_summary.csv"
        
        assert expected_png.exists(), "PNG 圖表檔案未建立"
        assert expected_csv.exists(), "CSV 摘要檔案未建立"
        
        # 4. 驗證 CSV 內容是否正確
        df = pd.read_csv(expected_csv)
        assert len(df) == 2, "CSV 應該有 2 筆驗證資料 (對應 2 個 epoch)"
        assert df.iloc[0]['Eval_Accuracy'] == 0.80
        assert df.iloc[1]['Eval_Accuracy'] == 0.90

    def test_empty_history_handling(self, tmp_path, capsys):
        """
        測試當 log_history 為空時，程式是否能正常處理而不崩潰
        """
        output_dir = tmp_path / "empty_viz"
        visualizer = TrainingVisualizer(str(output_dir))
        
        # 傳入空 list
        visualizer.plot([])
        
        # 檢查是否有印出警告
        captured = capsys.readouterr()
        assert "警告：Log history 為空" in captured.out
        
        # 確保沒有產生檔案
        assert not (output_dir / "training_metrics_visualization.png").exists()

    def test_ensure_dir_exists(self, tmp_path):
        """
        測試初始化時是否會自動建立目錄
        """
        new_dir = tmp_path / "nested" / "folder"
        assert not new_dir.exists()
        
        TrainingVisualizer(str(new_dir))
        
        assert new_dir.exists()
