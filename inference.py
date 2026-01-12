import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
from typing import Dict, List, Union, Optional
import os
import json
import sys

class RoBERTaInference:
    """
    RoBERTa Inference Engine for Knowledge Tracing
    """
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the inference engine.
        
        Args:
            model_path (str): Path to the directory containing the saved model and tokenizer.
            device (str, optional): 'cuda' or 'cpu'. Defaults to auto-detection.
        """
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self.max_len = 512
        self.label_map = {0: "待加強", 1: "尚可", 2: "精熟"}
        
        self.load_model()

    def load_model(self):
        """Load model and tokenizer from disk."""
        print(f"Loading model from {self.model_path}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded successfully on {self.device}.")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def _clean_performance_labels(self, text: str) -> str:
        """
        移除 Short_Answer_Log 中的學生表現標籤，保持與訓練時一致。
        
        移除格式：［學生表現］：Correct / Partially Correct / Incorrect
        """
        import re
        cleaned = re.sub(
            r'［學生表現］[:：]\s*(Correct|Partially Correct|Incorrect)\s*',
            '',
            text,
            flags=re.IGNORECASE
        )
        return cleaned.strip()

    def _format_text(self, sample: Dict) -> str:
        """
        Format input text to match training data format.
        
        Args:
            sample (Dict): Dictionary containing 'chapter', 'section', 'Short_Answer_Log', 'Dialog'.
            
        Returns:
            str: Formatted input string.
        """
        chapter = str(sample.get('chapter', ''))
        section = str(sample.get('section', ''))
        # 移除學生表現標籤，與訓練時保持一致
        short_answer_log = self._clean_performance_labels(str(sample.get('Short_Answer_Log', '')))
        dialog = str(sample.get('Dialog', ''))
        
        # Consistent with KTDynamicDataset
        formatted_text = (
            f"章節 : {chapter}\n"
            f"知識點 : {section}\n"
            f"學生掌握度 : [MASK]\n"
            f"簡答題作答紀錄 :\n{short_answer_log}\n"
            f"對話紀錄 :\n{dialog}\n"
        )
        return formatted_text

    def predict_single(self, sample: Dict) -> Dict:
        """
        Perform inference on a single sample.
        
        Args:
            sample (Dict): Input sample.
            
        Returns:
            Dict: Prediction result including label, id, confidence, and probabilities.
        """
        text = self._format_text(sample)
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_id = np.argmax(probs)
            
        return {
            "prediction": self.label_map[pred_id],
            "prediction_id": int(pred_id),
            "confidence": float(probs[pred_id]),
            "probabilities": {
                self.label_map[i]: float(probs[i]) for i in range(len(probs))
            }
        }

    def predict_batch(self, input_file: str, output_file: str = None) -> List[Dict]:
        """
        Perform batch inference from a CSV file.
        
        Args:
            input_file (str): Path to input CSV.
            output_file (str, optional): Path to save output CSV.
            
        Returns:
            List[Dict]: List of prediction results.
        """
        print(f"Reading data from {input_file}...")
        try:
            # Try utf-8-sig first (common for Excel CSVs), then utf-8
            df = pd.read_csv(input_file, encoding='utf-8-sig')
        except UnicodeDecodeError:
            df = pd.read_csv(input_file, encoding='utf-8')
            
        required_cols = ['chapter', 'section', 'Short_Answer_Log', 'Dialog']
        # Check if basic columns exist (allow missing if they can be empty strings, but headers must exist)
        # Actually, let's just use .get in loop, but warn if completely missing
        
        results = []
        print(f"Processing {len(df)} samples...")
        
        predictions = []
        confidences = []
        
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                print(f"Processed {idx}/{len(df)}")
                
            sample = row.to_dict()
            result = self.predict_single(sample)
            results.append(result)
            
            predictions.append(result['prediction'])
            confidences.append(result['confidence'])
            
        # Add results to dataframe
        df['Predicted_Mastery'] = predictions
        df['Confidence'] = confidences
        
        if output_file:
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"Results saved to {output_file}")
            
        return results

def main():
    parser = argparse.ArgumentParser(description="RoBERTa Inference CLI")
    parser.add_argument("--model", type=str, required=True, help="Path to the saved model directory")
    parser.add_argument("--input", type=str, help="Path to input CSV file for batch inference")
    parser.add_argument("--output", type=str, help="Path to output CSV file")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--format", type=str, choices=['json', 'csv', 'pretty'], default='pretty', help="Output format for single/interactive")
    
    args = parser.parse_args()
    
    if not args.input and not args.interactive:
        parser.error("Either --input or --interactive must be provided")
        
    engine = RoBERTaInference(args.model)
    
    if args.input:
        engine.predict_batch(args.input, args.output)
        
    if args.interactive:
        print("\n--- Interactive Inference Mode ---")
        print("Enter 'exit' to quit.\n")
        while True:
            try:
                chapter = input("Chapter: ")
                if chapter.lower() == 'exit': break
                
                section = input("Section: ")
                short_answer = input("Short Answer Log: ")
                dialog = input("Dialog: ")
                
                sample = {
                    "chapter": chapter,
                    "section": section,
                    "Short_Answer_Log": short_answer,
                    "Dialog": dialog
                }
                
                result = engine.predict_single(sample)
                
                if args.format == 'json':
                    print(json.dumps(result, indent=2, ensure_ascii=False))
                else:
                    print("\nPrediction Result:")
                    print(f"  Outcome: {result['prediction']}")
                    print(f"  Confidence: {result['confidence']:.2%}")
                    print(f"  Probabilities: {json.dumps(result['probabilities'], ensure_ascii=False)}")
                    print("-" * 30)
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()
