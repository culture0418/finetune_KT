from inference import RoBERTaInference
import os
import sys

# Add parent directory to path so we can import inference.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    # Path to the trained model
    # Note: Using the path found in the previous step
    model_path = "results/roberta-chinese_20260112_201404/final_model/"
    
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist.")
        return

    print("Initializing Inference Engine...")
    engine = RoBERTaInference(model_path)

    # Sample data matching the actual format
    sample = {
        "chapter": "A_機器學習-監督式學習",
        "section": "人工智慧的起源",
        "Short_Answer_Log": """［題目］：（簡答題）艾倫．圖靈（Alan Turing）於 1950 年提出的「圖靈測試」，其判斷電腦是否具備智慧的標準是什麼？
［學生答案］：電腦能否透過文字達成無法分辨的人類表達方式與智慧
［學生表現］：Partially Correct
［題目］：（簡答題）根據「AI 發展簡史」圖表，人工智慧的發展歷程主要被劃分為哪三個時期
［學生答案］：推理期、知識期、學習期
［學生表現］：Correct""",
        "Dialog": """[課後對話]
[學生]: 偏置Bias是什麼?
[AI Tutor]: 偏置（Bias）是神經網絡中的一個重要參數...
[學生]: 激活函數和偏置Bias的差異
[AI Tutor]: 激活函數和偏置（Bias）在神經網絡中..."""
    }

    print("\nPredicting on sample...")
    result = engine.predict_single(sample)
    
    print("\nResult:")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("Probabilities:")
    for label, prob in result['probabilities'].items():
        print(f"  {label}: {prob:.4f}")

if __name__ == "__main__":
    main()
