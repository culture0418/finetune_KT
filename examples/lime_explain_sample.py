from lime_explainer import LIMEExplainer
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    model_path = "results/roberta-chinese_20260111_034253/final_model/"
    
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist.")
        return

    print("Initializing LIME Explainer with LLM keyword extraction...")
    explainer = LIMEExplainer(model_path, use_llm_keywords=True)

    # Sample data
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

    # Collect all keywords for full report (avoid duplicate API calls)
    all_keywords = []

    # Test Focus 1: Student Answers (with LLM keywords)
    print("\n--- Explaining Student Answers (LLM Keywords) ---")
    exp_answers, keywords_answers = explainer.explain_prediction_with_keywords(
        sample, 
        focus_on="student_answers", 
        num_features=10, 
        num_samples=200
    )
    if exp_answers:
        explainer.generate_html_report(
            exp_answers, 
            "lime_reports/explanation_answers_llm.html",
            original_text=sample["Short_Answer_Log"],
            keywords=keywords_answers,
            title="學生回答分析"
        )
    if keywords_answers:
        all_keywords.extend(keywords_answers)

    # Test Focus 2: Student Questions (with LLM keywords)
    print("\n--- Explaining Student Questions (LLM Keywords) ---")
    exp_questions, keywords_questions = explainer.explain_prediction_with_keywords(
        sample, 
        focus_on="student_questions", 
        num_features=10, 
        num_samples=200
    )
    if exp_questions:
        explainer.generate_html_report(
            exp_questions, 
            "lime_reports/explanation_questions_llm.html",
            original_text=sample["Dialog"],
            keywords=keywords_questions,
            title="學生提問分析"
        )
    if keywords_questions:
        all_keywords.extend(keywords_questions)

    # Test Focus 3: Full Text (using combined keywords, no extra API call)
    print("\n--- Explaining Full Text (Combined Keywords) ---")
    if all_keywords:
        # Remove duplicates while preserving order
        unique_keywords = list(dict.fromkeys(all_keywords))
        print(f"Using combined keywords: {unique_keywords}")
        exp_full = explainer.explain_with_keywords(
            sample, 
            keywords=unique_keywords,
            num_features=15, 
            num_samples=200
        )
        if exp_full:
            # Combine all text for full report
            full_text = f"""【章節】{sample['chapter']}
【知識點】{sample['section']}

【簡答題記錄】
{sample['Short_Answer_Log']}

【對話記錄】
{sample['Dialog']}"""
            explainer.generate_html_report(
                exp_full, 
                "lime_reports/explanation_full_llm.html",
                original_text=full_text,
                keywords=unique_keywords,
                title="完整文本分析"
            )

if __name__ == "__main__":
    main()
