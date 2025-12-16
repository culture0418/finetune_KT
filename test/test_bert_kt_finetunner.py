import pytest
import pandas as pd
import torch
from transformers import TrainingArguments, BertTokenizer, BertConfig, BertForSequenceClassification
from finetune_bert import BertKTFinetuner, KTDataProcessor, KTDynamicDataset

class DummyTrainer:
    """用於 mock Hugging Face Trainer 行為"""
    def __init__(self, *args, **kwargs):
        self.model = kwargs.get('model')
        self.state = type('State', (), {'log_history': [
            {"epoch": 1, "eval_accuracy": 0.7},
            {"epoch": 2, "eval_accuracy": 0.8},
            {"epoch": 3, "eval_accuracy": 0.9}
        ]})()
    def train(self):
        class Output:
            metrics = {"train_runtime": 1.0}
        return Output()
    def evaluate(self):
        return {"eval_accuracy": 0.9}
    def save_model(self, save_path):
        self.saved_path = save_path

@pytest.fixture
def sample_dataframe():
    # 測試資料擴充為 12 筆，四種標籤均衡分布
    return pd.DataFrame({
        'chapter': [
            '監督式學習', '監督式學習', '監督式學習',
            '深度學習', '深度學習', '深度學習',
            '非監督式學習', '非監督式學習', '非監督式學習',
            '模型評估', '模型評估', '模型評估'
        ],
        'section': [
            '線性迴歸', '分類', '回歸',
            '神經網路', '卷積神經網路', '遞迴神經網路',
            'K-means', 'PCA', '異常檢測',
            '混淆矩陣', 'ROC曲線', 'F1-Score'
        ],
        'all_logs': [
            '答對5題答錯1題', '答對3題答錯2題', '答對2題答錯3題',
            '答對4題答錯2題', '答對1題答錯4題', '答對0題答錯5題',
            '答對3題答錯3題', '答對2題答錯4題', '答對1題答錯5題',
            '答對5題答錯0題', '答對4題答錯1題', '答對3題答錯2題'
        ],
        'Preview_ChatLog': [
            '這是什麼概念', '分類怎麼做', '回歸有什麼用',
            '原理是什麼', 'CNN是什麼', 'RNN怎麼運作',
            '聚類分析怎麼做', 'PCA怎麼用', '異常怎麼判斷',
            'TP和FP是什麼', 'ROC是什麼', 'F1怎麼算'
        ],
        'Review_ChatLog': [
            '我懂了', '還需要練習', '完全理解',
            '我不太懂', '需要加強', '完全不懂',
            '還可以', '再複習', '不太清楚',
            '明白了', '理解了', '掌握了'
        ],
        'Mastery_Level_K4': [
            '精熟', '尚可', '待加強',
            '良好', '尚可', '待加強',
            '精熟', '尚可', '待加強',
            '精熟', '良好', '尚可'
        ],
        'labels': [3, 1, 0, 2, 1, 0, 3, 1, 0, 3, 2, 1]
    })

@pytest.fixture
def processor(tmp_path, sample_dataframe):
    # 建立暫存 CSV
    csv_path = tmp_path / "test.csv"
    sample_dataframe.to_csv(csv_path, index=False)
    proc = KTDataProcessor(str(csv_path))
    proc.prepare_data(test_size=0.5, random_state=42)
    return proc

@pytest.fixture
def training_args():
    return TrainingArguments(
        output_dir="./dummy_output",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        logging_dir="./dummy_logs",
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none"
    )

@pytest.fixture
def finetuner(monkeypatch, processor, training_args):
    # 強制使用 bert-base-chinese
    model_name = "bert-base-chinese"
    # Patch Trainer 為 DummyTrainer
    monkeypatch.setattr("finetune_bert.Trainer", DummyTrainer)
    return BertKTFinetuner(model_name, processor, training_args)

def test_init(monkeypatch, processor, training_args):
    """測試 __init__ 初始化流程"""
    model_name = "bert-base-chinese"
    # Patch Trainer 為 DummyTrainer
    monkeypatch.setattr("finetune_bert.Trainer", DummyTrainer)
    finetuner = BertKTFinetuner(model_name, processor, training_args)
    assert finetuner.model_name == model_name
    assert finetuner.processor == processor
    assert finetuner.training_args == training_args
    assert isinstance(finetuner.tokenizer, BertTokenizer)
    assert isinstance(finetuner.model, BertForSequenceClassification)
    assert finetuner.model.config.num_labels == 4
    assert finetuner.model.config.id2label == {0: "待加強", 1: "尚可", 2: "良好", 3: "精熟"}
    assert finetuner.model.config.label2id == {"待加強": 0, "尚可": 1, "良好": 2, "精熟": 3}
    assert finetuner.trainer is None

def test_compute_metrics():
    """測試 _compute_metrics 靜態方法"""
    logits = [[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]]
    labels = [2, 0]
    result = BertKTFinetuner._compute_metrics((logits, labels))
    assert "accuracy" in result
    assert 0.5 <= result["accuracy"] <= 1.0

def test_run_finetuning(monkeypatch, finetuner):
    """測試 run_finetuning 執行流程（含 accuracy 曲線）"""
    # Patch matplotlib.pyplot.show 以避免阻塞
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    finetuner.run_finetuning()
    # 應該初始化 trainer
    assert isinstance(finetuner.trainer, DummyTrainer)
    # 應該能取得 evaluate 結果
    eval_result = finetuner.trainer.evaluate()
    assert "eval_accuracy" in eval_result

def test_save_model(finetuner, tmp_path):
    """測試 save_model 儲存流程"""
    # 先初始化 trainer
    finetuner.trainer = DummyTrainer(model=finetuner.model)
    save_path = tmp_path / "saved_model"
    finetuner.save_model(str(save_path))
    # 檢查 DummyTrainer 是否記錄儲存路徑
    assert finetuner.trainer.saved_path == str(save_path)

def test_save_model_without_train(finetuner):
    """測試 save_model 未訓練時的例外處理"""
    finetuner.trainer = None
    with pytest.raises(RuntimeError, match="模型尚未訓練"):
        finetuner.save_model("dummy_path")
