import os
import re
import json
import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer
from inference import RoBERTaInference
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

class LIMEExplainer:
    """
    LIME Explainer for RoBERTa Knowledge Tracing Model
    Supports targeted analysis using LLM-extracted keywords.
    """
    def __init__(self, model_path: str, use_llm_keywords: bool = True):
        """
        Initialize the LIME explainer.
        
        Args:
            model_path (str): Path to the trained model.
            use_llm_keywords (bool): Whether to use LLM for keyword extraction.
        """
        self.inference_engine = RoBERTaInference(model_path)
        self.class_names = ["待加強", "尚可", "精熟"]
        self.use_llm_keywords = use_llm_keywords
        
        # Initialize OpenAI client if LLM keywords enabled
        if self.use_llm_keywords:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("Warning: OPENAI_API_KEY not found in .env. LLM keyword extraction disabled.")
                self.use_llm_keywords = False
            else:
                self.openai_client = OpenAI(api_key=api_key)
                self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                print(f"LLM keyword extraction enabled (model: {self.openai_model})")

    def _extract_keywords_with_llm(self, text: str, focus_type: str) -> List[str]:
        """
        Use LLM to extract meaningful keywords from text.
        
        Args:
            text: The text to extract keywords from.
            focus_type: Type of content ('student_answers' or 'student_questions').
            
        Returns:
            List of extracted keywords.
        """
        if not self.use_llm_keywords or not text.strip():
            return []
            
        if focus_type == "student_answers":
            prompt = f"""請從以下學生簡答題回答中提取 5-10 個最重要的專業術語或核心概念詞彙。

重要指引：
1. **只提取**：專業術語、技術名詞、核心概念
2. **必須排除**：疑問詞（什麼、如何、為何）、語氣詞、連接詞、介詞、代詞
3. 關鍵詞應該是名詞或名詞短語，能反映學生理解的專業知識點

學生回答：
{text}

請以 JSON 陣列格式輸出，例如：["機器學習", "監督式學習", "神經網絡"]
只輸出 JSON，不要其他文字。"""
        else:  # student_questions
            prompt = f"""請從以下學生提問中提取 5-10 個最重要的專業術語或核心概念詞彙。

重要指引：
1. **只提取**：專業術語、技術名詞、核心概念
2. **必須排除**：疑問詞（什麼、如何、為何、哪裡、誰）、語氣詞、連接詞、介詞、代詞
3. 關鍵詞應該是名詞或名詞短語，能反映學生疑問的專業知識點

學生提問：
{text}

請以 JSON 陣列格式輸出，例如：["偏置", "Bias", "激活函數"]
只輸出 JSON，不要其他文字。"""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )
            
            content = response.choices[0].message.content.strip()
            print(f"LLM raw response: {content[:200]}...")  # Debug
            
            # Try to extract JSON array from response
            # Handle cases where LLM adds extra text
            if content.startswith('['):
                keywords = json.loads(content)
            else:
                # Try to find JSON array in response
                match = re.search(r'\[.*?\]', content, re.DOTALL)
                if match:
                    keywords = json.loads(match.group())
                else:
                    print(f"Warning: No JSON array found in response")
                    return []
                    
            print(f"LLM extracted keywords: {keywords}")
            return keywords
            
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse LLM response as JSON: {e}")
            return []
        except Exception as e:
            print(f"Warning: LLM keyword extraction failed: {e}")
            return []
        
    def _extract_focus_text(self, text_data: Dict, focus_on: str) -> str:
        """
        Extract specific text segment for analysis based on focus_on parameter.
        """
        if focus_on == "student_answers":
            short_answer_log = text_data.get("Short_Answer_Log", "")
            # Extract only student answers, excluding performance labels
            # Pattern: ［學生答案］：... (until next ［ or end)
            student_answers = re.findall(
                r'［學生答案］[:：]\s*(.+?)(?=［|$)', 
                short_answer_log, 
                re.DOTALL
            )
            return " | ".join([ans.strip() for ans in student_answers]) if student_answers else short_answer_log
        
        elif focus_on == "student_questions":
            dialog = text_data.get("Dialog", "")
            student_questions = re.findall(
                r'\[學生\][:：]\s*(.+?)(?=\[AI Tutor\]|\[學生\]|$)', 
                dialog, 
                re.DOTALL
            )
            return " ".join([q.strip() for q in student_questions])
            
        elif focus_on == "dialog_student":
             return self._extract_focus_text(text_data, "student_questions")

        elif focus_on == "dialog_tutor":
            dialog = text_data.get("Dialog", "")
            tutor_responses = re.findall(
                r'\[AI Tutor\][:：]\s*(.+?)(?=\[學生\]|\[AI Tutor\]|$)', 
                dialog, 
                re.DOTALL
            )
            return " ".join([resp.strip() for resp in tutor_responses])
            
        elif focus_on == "dialog_all":
            return text_data.get("Dialog", "")
            
        else: # full_text
            return self.inference_engine._format_text(text_data)

    def _reconstruct_sample(self, text_data: Dict, focus_on: str, perturbed_text: str) -> Dict:
        """
        Reconstruct the full sample with the perturbed text segment.
        """
        new_sample = text_data.copy()
        
        if focus_on == "student_answers":
             new_sample['Short_Answer_Log'] = f"［學生答案］：{perturbed_text}"
             
        elif focus_on == "student_questions":
            new_sample['Dialog'] = f"[學生]: {perturbed_text}"
            
        elif focus_on == "dialog_student":
            new_sample['Dialog'] = f"[學生]: {perturbed_text}"
            
        elif focus_on == "dialog_tutor":
            new_sample['Dialog'] = f"[AI Tutor]: {perturbed_text}"
            
        elif focus_on == "dialog_all":
            new_sample['Dialog'] = perturbed_text
        
        return new_sample

    def explain_prediction(self, text_data: Dict, focus_on: str = "full_text", 
                          num_features: int = 10, num_samples: int = 1000):
        """
        Run LIME explanation with LLM-extracted keywords.
        """
        exp, _ = self.explain_prediction_with_keywords(text_data, focus_on, num_features, num_samples)
        return exp

    def explain_prediction_with_keywords(self, text_data: Dict, focus_on: str = "full_text", 
                          num_features: int = 10, num_samples: int = 1000):
        """
        Run LIME explanation and return both explanation and extracted keywords.
        Returns: (explanation, keywords)
        """
        # 1. Get text to explain
        if focus_on == "full_text":
            text_to_explain = self.inference_engine._format_text(text_data)
            keywords = None
        else:
            text_to_explain = self._extract_focus_text(text_data, focus_on)
            if self.use_llm_keywords:
                keywords = self._extract_keywords_with_llm(text_to_explain, focus_on)
            else:
                keywords = None
            
        if not text_to_explain.strip():
            print(f"Warning: Extracted text for '{focus_on}' is empty.")
            return None, None

        # 2. If we have LLM keywords, use them
        if keywords:
            # Use a unique delimiter to keep multi-word keywords together (e.g., "Partially Correct")
            delimiter = "|||"
            text_to_explain = delimiter.join(keywords)
            print(f"Using LLM keywords for LIME: {keywords}")
            split_fn = lambda x: x.split(delimiter)
        else:
            split_fn = list

        # 3. Define classifier function
        def classifier_fn(texts):
            if focus_on == "full_text":
                full_texts = texts
            else:
                full_texts = []
                for t in texts:
                    s = self._reconstruct_sample(text_data, focus_on, t)
                    full_texts.append(self.inference_engine._format_text(s))
            
            return self._predict_raw(full_texts)

        # 4. Create Explainer
        explainer = LimeTextExplainer(
            class_names=self.class_names,
            split_expression=split_fn
        )
        
        # 5. Explain
        exp = explainer.explain_instance(
            text_to_explain,
            classifier_fn,
            num_features=num_features,
            num_samples=num_samples,
            top_labels=3  # Explain top 3 labels
        )
        return exp, keywords

    def explain_with_keywords(self, text_data: Dict, keywords: List[str],
                             num_features: int = 10, num_samples: int = 1000):
        """
        Run LIME explanation using pre-extracted keywords (no API call).
        """
        if not keywords:
            print("Warning: No keywords provided.")
            return None
        
        # Use delimiter to keep multi-word keywords together
        delimiter = "|||"
        text_to_explain = delimiter.join(keywords)
        print(f"Using provided keywords for LIME: {keywords}")
        split_fn = lambda x: x.split(delimiter)

        def classifier_fn(texts):
            # For combined keywords, we reconstruct with all content
            full_texts = []
            for t in texts:
                # Create a sample with the perturbed keywords as the main content
                new_sample = text_data.copy()
                new_sample['Short_Answer_Log'] = f"［學生答案］：{t}"
                new_sample['Dialog'] = f"[學生]: {t}"
                full_texts.append(self.inference_engine._format_text(new_sample))
            return self._predict_raw(full_texts)

        explainer = LimeTextExplainer(
            class_names=self.class_names,
            split_expression=split_fn
        )
        
        exp = explainer.explain_instance(
            text_to_explain,
            classifier_fn,
            num_features=num_features,
            num_samples=num_samples,
            top_labels=3
        )
        return exp

    def _predict_raw(self, texts: List[str]) -> np.ndarray:
        """
        Raw prediction helper.
        """
        import torch
        
        encoding = self.inference_engine.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self.inference_engine.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding['input_ids'].to(self.inference_engine.device)
        attention_mask = encoding['attention_mask'].to(self.inference_engine.device)
        
        batch_size = 16
        probs_list = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_ids = input_ids[i:i+batch_size]
                batch_mask = attention_mask[i:i+batch_size]
                
                outputs = self.inference_engine.model(input_ids=batch_ids, attention_mask=batch_mask)
                logits = outputs.logits
                batch_probs = torch.softmax(logits, dim=1).cpu().numpy()
                probs_list.append(batch_probs)
                
        return np.concatenate(probs_list, axis=0)

    def _get_student_text_regions(self, text: str) -> List[tuple]:
        """
        Identify regions of text that contain student content (answers and questions).
        
        Args:
            text: Full text containing student answers and questions
            
        Returns:
            List of (start, end) tuples marking student text regions
        """
        regions = []
        
        # Find all [學生答案] regions
        for match in re.finditer(r'［學生答案］[:：]\s*(.+?)(?=［|$)', text, re.DOTALL):
            regions.append((match.start(1), match.end(1)))
        
        # Find all [學生] question regions  
        for match in re.finditer(r'\[學生\][:：]\s*(.+?)(?=\[AI Tutor\]|\[學生\]|$)', text, re.DOTALL):
            regions.append((match.start(1), match.end(1)))
        
        # Sort by start position
        regions.sort(key=lambda x: x[0])
        return regions

    def generate_html_report(self, exp, output_path: str, original_text: str = None, 
                             keywords: List[str] = None, title: str = "LIME Explanation"):

        """
        Save explanation as HTML with full text and keyword highlighting.
        
        Args:
            exp: LIME explanation object
            original_text: Full original text to display with highlights
            keywords: List of keywords to highlight in the original text
            title: Report title
        """
        if exp is None: 
            return
        
        # Get feature weights from explanation
        if exp.top_labels is None or len(exp.top_labels) == 0:
            print("Warning: No labels in explanation")
            return
        pred_label = exp.top_labels[0]
        feature_weights = dict(exp.as_list(label=pred_label))
        
        # Get prediction probabilities
        probs = exp.predict_proba
        prob_html = ""
        for i, (label, prob) in enumerate(zip(self.class_names, probs)):
            color = "#22c55e" if i == pred_label else "#6b7280"
            prob_html += f'<div style="margin: 5px 0;"><span style="color: {color}; font-weight: {"bold" if i == pred_label else "normal"};">{label}: {prob:.2%}</span></div>'
        
        # Create highlighted text using robust index-based replacement
        highlighted_text = ""
        if original_text and keywords:
            # Get student text regions (where we want highlighting)
            student_regions = self._get_student_text_regions(original_text)
            
            # Helper function to check if a position is in student text
            def is_in_student_region(pos):
                return any(start <= pos < end for start, end in student_regions)
            
            # 1. Find all matches for all keywords ONLY in student text regions
            # List of tuples: (start_index, end_index, keyword, weight)
            all_matches = []
            
            for keyword in keywords:
                if keyword not in feature_weights:
                    continue
                    
                weight = feature_weights[keyword]
                # Find all occurrences of this keyword in the text
                # Use regex escape to handle special characters safely
                for match in re.finditer(re.escape(keyword), original_text, re.IGNORECASE):
                    # Only add if match is within student text regions
                    if is_in_student_region(match.start()):
                        all_matches.append({
                            "start": match.start(),
                            "end": match.end(),
                            "keyword": match.group(), # Use exact text from match to preserve case
                            "weight": weight,
                            "length": match.end() - match.start()
                        })
            
            # 2. Resolve overlaps: Prioritize longer keywords (e.g., "Partially Correct" over "Correct")
            # Sort by length (descending) then by start position
            all_matches.sort(key=lambda x: (x['length'], -x['start']), reverse=True)
            
            # Keep track of occupied indices
            occupied = [False] * len(original_text)
            final_matches = []
            
            for match in all_matches:
                # Check if this range is already occupied
                is_occupied = any(occupied[i] for i in range(match['start'], match['end']))
                if not is_occupied:
                    # Mark as occupied and keep match
                    for i in range(match['start'], match['end']):
                        occupied[i] = True
                    final_matches.append(match)
            
            # 3. Construct HTML by processing matches in order
            # Sort matches by start position for reconstruction
            final_matches.sort(key=lambda x: x['start'])
            
            last_idx = 0
            segments = []
            
            for match in final_matches:
                # Add non-highlighted text before this match
                segments.append(original_text[last_idx:match['start']])
                
                # Add highlighted text
                weight = match['weight']
                match_text = original_text[match['start']:match['end']]
                
                if weight > 0:
                    # Positive - Green
                    bg_color = f"rgba(34, 197, 94, {min(abs(weight) * 2, 0.8)})"
                else:
                    # Negative - Red
                    bg_color = f"rgba(239, 68, 68, {min(abs(weight) * 2, 0.8)})"
                
                span = f'<span style="background-color: {bg_color}; padding: 2px 4px; border-radius: 3px; border: 1px solid rgba(0,0,0,0.1);" title="Weight: {weight:.4f}">{match_text}</span>'
                segments.append(span)
                
                last_idx = match['end']
            
            # Add remaining text
            segments.append(original_text[last_idx:])
            highlighted_text = "".join(segments)
        
        # Build feature importance table
        feature_rows = ""
        sorted_features = sorted(feature_weights.items(), key=lambda x: abs(x[1]), reverse=True)
        for feature, weight in sorted_features:
            color = "#22c55e" if weight > 0 else "#ef4444"
            bar_width = min(abs(weight) * 200, 150)
            direction = "right" if weight > 0 else "left"
            feature_rows += f'''
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #e5e7eb;">{feature}</td>
                <td style="padding: 8px; border-bottom: 1px solid #e5e7eb; text-align: right;">
                    <span style="color: {color};">{weight:+.4f}</span>
                </td>
                <td style="padding: 8px; border-bottom: 1px solid #e5e7eb; width: 200px;">
                    <div style="width: {bar_width}px; height: 16px; background-color: {color}; 
                                float: {"left" if weight > 0 else "right"}; border-radius: 3px;"></div>
                </td>
            </tr>'''
        
        # Generate complete HTML
        html_content = f'''<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; 
               max-width: 1200px; margin: 0 auto; padding: 20px; background: #f9fafb; }}
        .card {{ background: white; border-radius: 12px; padding: 24px; margin-bottom: 20px; 
                box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        h1 {{ color: #1f2937; margin-bottom: 8px; }}
        h2 {{ color: #374151; border-bottom: 2px solid #e5e7eb; padding-bottom: 8px; }}
        .prediction {{ font-size: 24px; font-weight: bold; color: #059669; }}
        .original-text {{ background: #f3f4f6; padding: 16px; border-radius: 8px; 
                         line-height: 1.8; white-space: pre-wrap; font-size: 14px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th {{ text-align: left; padding: 12px 8px; background: #f9fafb; border-bottom: 2px solid #e5e7eb; }}
        .legend {{ display: flex; gap: 20px; margin-top: 10px; }}
        .legend-item {{ display: flex; align-items: center; gap: 8px; }}
        .legend-box {{ width: 20px; height: 20px; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="card">
        <h1>🔍 {title}</h1>
        <p style="color: #6b7280;">模型可解釋性分析報告</p>
    </div>
    
    <div class="card">
        <h2>📊 預測結果</h2>
        <div class="prediction">{self.class_names[pred_label]}</div>
        <div style="margin-top: 16px;">
            <strong>各類別機率：</strong>
            {prob_html}
        </div>
    </div>
    
    {"<div class='card'><h2>📝 原始文本（關鍵字高亮）</h2><div class='legend'><div class='legend-item'><div class='legend-box' style='background: rgba(34, 197, 94, 0.5);'></div><span>正向影響（支持預測）</span></div><div class='legend-item'><div class='legend-box' style='background: rgba(239, 68, 68, 0.5);'></div><span>負向影響（反對預測）</span></div></div><div class='original-text' style='margin-top: 16px;'>" + highlighted_text + "</div></div>" if highlighted_text else ""}
    
    <div class="card">
        <h2>📈 特徵重要性</h2>
        <table>
            <thead>
                <tr>
                    <th>關鍵詞</th>
                    <th style="text-align: right;">權重</th>
                    <th>影響程度</th>
                </tr>
            </thead>
            <tbody>
                {feature_rows}
            </tbody>
        </table>
    </div>
    
    <div class="card" style="text-align: center; color: #9ca3af; font-size: 12px;">
        Generated by LIME Explainer | {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
    </div>
</body>
</html>'''
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Report saved to {output_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--focus", default="student_answers")
    parser.add_argument("--output", default="lime_report.html")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM keyword extraction")
    args = parser.parse_args()
    
    explainer = LIMEExplainer(args.model, use_llm_keywords=not args.no_llm)
    
    sample = {
        "chapter": "測試章節",
        "section": "測試知識點",
        "Short_Answer_Log": "［學生答案］：電腦能否透過文字達成無法分辨的人類表達方式與智慧",
        "Dialog": "[學生]: 偏置Bias是什麼?"
    }
    
    print(f"Generating LIME explanation for focus: {args.focus}...")
    exp = explainer.explain_prediction(sample, focus_on=args.focus, num_samples=100)
    explainer.generate_html_report(exp, args.output)

if __name__ == "__main__":
    main()
