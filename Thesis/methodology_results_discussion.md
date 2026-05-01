# 第三章　研究方法

## 3.1 整體研究架構

本研究以「自動化判斷學生對知識點之掌握等級」為核心任務，透過微調中文 RoBERTa 模型並與多個大型語言模型（LLM）進行對照實驗，評估不同模型在此分類任務上的表現。整體流程如下：

1. **資料準備**：將原始作答記錄資料集進行清理與三段式切分（train/val/test）。
2. **RoBERTa 微調**：使用 Optuna 進行超參數搜索，並以加權交叉熵損失與早停策略訓練最終模型。
3. **LLM 評測**：以零樣本（zero-shot）方式提示 17 個 LLM API，產生預測結果。
4. **公平比較**：所有模型統一在 held-out test set 上評估，計算各項指標。

---

## 3.2 資料集

### 3.2.1 資料來源
本研究使用 `finetune_dataset_1142_v4_without_chat_0227.csv`，內容為學生在不同知識點上的簡答題作答記錄。每筆資料包含以下欄位：

| 欄位名稱 | 說明 |
|---|---|
| `user_id`, `username` | 學生識別資訊（不作為模型輸入） |
| `chapter` | 章節名稱 |
| `section` | 知識點名稱 |
| `Short_Answer_Log` | 該知識點下所有簡答題之題目、參考答案、學生答案、per-item 評分等 |
| `Mastery_Ratio` | 掌握度比例（連續值） |
| `Mastery_Label` | 掌握度等級（**分類目標**） |

### 3.2.2 標籤定義
`Mastery_Label` 採三級分類，數值映射如下：

| Label | 數值 | 意義 |
|---|---|---|
| 待加強 | 0 | 整體掌握度低 |
| 尚可 | 1 | 部分掌握 |
| 精熟 | 2 | 完全掌握 |

### 3.2.3 資料規模與分佈
經清理後共 **474** 筆有效資料。標籤分佈為：

| 類別 | 筆數 | 比例 |
|---|---|---|
| 待加強 | 47 | 9.92% |
| 尚可 | 337 | 71.10% |
| 精熟 | 90 | 18.99% |

⚠️ 資料呈現高度類別不平衡（尚可佔 71%），後續訓練必須處理此問題。

---

## 3.3 資料切分

### 3.3.1 切分策略
為確保 RoBERTa 與 LLM 之比較公平性，本研究採用 **70/15/15** 比例之 stratified 切分：

- **Train set (70%)**：用於 RoBERTa fine-tuning。
- **Validation set (15%)**：用於 Optuna 目標函數計算與訓練期早停。
- **Test set (15%)**：對 RoBERTa 訓練完全 held-out，作為 RoBERTa 與所有 LLM 之共同最終評測集。

### 3.3.2 切分實作
使用兩段式 stratified split（依 `Mastery_Label` 分層抽樣，`random_state=42`）：

1. 首先以 15% 切出 test set；
2. 從剩餘 85% 中再切出 ~17.65% 作為 val set，使三者比例對應 70/15/15。

切分結果固定為三個 CSV 檔案（`train.csv` / `val.csv` / `test.csv`），並隨 commit 進入版本控制，確保結果可重現。

### 3.3.3 切分結果

| Split | 筆數 | 比例 | 待加強 | 尚可 | 精熟 |
|---|---|---|---|---|---|
| Train | 331 | 69.83% | 33 (10.0%) | 235 (71.0%) | 63 (19.0%) |
| Val | 71 | 14.98% | 7 (9.9%) | 51 (71.8%) | 13 (18.3%) |
| Test | 72 | 15.19% | 7 (9.7%) | 51 (70.8%) | 14 (19.4%) |

各 split 之類別比例與整體分佈一致，stratification 成功。

---

## 3.4 RoBERTa 微調方法

### 3.4.1 預訓練模型
採用 **`hfl/chinese-roberta-wwm-ext`**（哈工大訊飛聯合實驗室訓練之中文 RoBERTa with Whole Word Masking），共 12 層 Transformer。下游接 `AutoModelForSequenceClassification` 之線性分類頭，輸出維度為 3。

### 3.4.2 輸入處理
將 `chapter`、`section`、`Short_Answer_Log` 三欄拼接為單一文字輸入，由 `BertTokenizer` 進行 tokenization，最大長度設為 512，採 `padding="max_length"` 並啟用 truncation。

### 3.4.3 處理類別不平衡：加權交叉熵
為緩解類別不平衡，自定義 `WeightedTrainer` 子類覆寫 `compute_loss`，使用基於訓練集統計的類別權重：

$$ w_i = \frac{N}{C \cdot n_i} $$

其中 $N$ 為訓練集樣本總數，$C=3$ 為類別數，$n_i$ 為類別 $i$ 之樣本數。實際權重：

| 類別 | 樣本數 | 權重 |
|---|---|---|
| 待加強 | 33 | 3.34 |
| 尚可 | 235 | 0.47 |
| 精熟 | 63 | 1.75 |

### 3.4.4 Optuna 超參數搜索

採用 Optuna 進行 15 trials 之超參數搜索，剪枝策略為 `MedianPruner(n_startup_trials=3, n_warmup_steps=5)`，目標函數為 val set 上的 macro-F1 (越高越好)。

**搜索空間**：

| 超參數 | 範圍 / 候選 |
|---|---|
| `learning_rate` | $[1\times10^{-5}, 1\times10^{-4}]$，log-uniform |
| `num_train_epochs` | $\{5, 10, 15\}$ |
| `per_device_train_batch_size` | $\{4, 8, 16\}$ |
| `warmup_steps` | $\{50, 100, 150, 200\}$ |
| `weight_decay` | $[0.0, 0.1]$，uniform |

15 trials 中 9 個 COMPLETE，6 個 PRUNED，總搜索時間約 8 分鐘（NVIDIA RTX 4090）。最佳 trial（#3）val macro-F1 為 0.9678，超參數為：

| 超參數 | 最佳值 |
|---|---|
| learning_rate | $3.888 \times 10^{-5}$ |
| num_train_epochs | 5 |
| per_device_train_batch_size | 8 |
| warmup_steps | 50 |
| weight_decay | 0.0215 |

### 3.4.5 完整訓練設定

為避免過擬合與提升訓練穩定性，於 Optuna 最佳超參數上做兩處保守調整：

| 項目 | Optuna 值 | 完整訓練值 | 動機 |
|---|---|---|---|
| `learning_rate` | $3.888 \times 10^{-5}$ | × 0.5 = $1.944 \times 10^{-5}$ | 降低避免 overshoot |
| `warmup_steps` | 50 | + 100 = 150 | 加長 warmup |
| `num_train_epochs` | 5 | 15 | 配合早停 |

其他關鍵設定：
- `eval_strategy` / `save_strategy` 均設為 `epoch`。
- `load_best_model_at_end=True`，`metric_for_best_model="macro_f1"`，自動載回 val 最佳 checkpoint。
- 早停：`EarlyStoppingCallback(patience=3)`。

### 3.4.6 Multi-Seed Evaluation Protocol

為避免單次訓練結果受隨機初始化、batch shuffle 順序與 GPU 浮點非確定性影響，本研究在最終訓練階段採用 **multi-seed 評估**，以五個不同 random seed 重複訓練：

$$\text{seeds} = \{42, 7, 123, 2024, 999\}$$

每個 seed 共享相同切分資料、相同最佳超參數（取自 §3.4.5）、相同保守調整策略，僅 `seed` 與 `data_seed` 不同。每次訓練完成後將最佳 checkpoint 載入並對 test set 推論，得到該 seed 的指標。最終回報以 mean ± std（n=5）形式呈現，量化模型訓練的穩定性。此做法符合 NeurIPS / ACL 等學術會議的標準作法。

---

## 3.5 LLM 比較設計

### 3.5.1 比較模型清單

涵蓋 17 個 LLM API，依供應商分類如下：

| 供應商 | 模型 |
|---|---|
| OpenAI | GPT-4o、GPT-4o-mini、GPT-4.1、o4-mini |
| Google Gemini | Gemini 2.5 Pro / Flash / Flash-Lite、Gemini 3.1 Pro / 3 Flash / 3.1 Flash-Lite |
| Google Gemma 3（open-weight） | Gemma 3 4B / 12B / 27B Instruct |
| Anthropic | Claude Sonnet 4.5 |
| Meta（via Groq） | Llama 3.3 70B、Llama 3.1 8B |
| Alibaba（via Groq） | Qwen3 32B |

### 3.5.2 提示模板（Prompt）
所有 LLM 統一使用相同 prompt：

```
[System]
你是一個教育評估專家。根據以下學生的學習資料，判斷該學生對此知識點的掌握程度。
請只回答以下三個等級之一：待加強、尚可、精熟

[User]
章節：{chapter}
知識點：{section}
簡答題作答紀錄：
{short_answer_log}

請只回答一個詞：待加強、尚可、或精熟
```

### 3.5.3 推論設定
- `temperature = 0`（除 reasoning 模型不支援者改用最低 reasoning effort）。
- 最大輸出長度設為足以涵蓋一個分類詞（多數模型 64-128 tokens；具 thinking 模式之 Gemini 2.5/3.x Pro 設為 1024 並過濾 thought=True 部分）。
- 失敗自動重試 3 次（指數退避）。
- 中斷續跑：每 10 筆儲存中間結果。

### 3.5.4 輸出解析
LLM 回應透過 regex 解析：先去除 `<think>...</think>` 區塊（適用於 Qwen3、Gemini reasoning），再依字串包含關係匹配三個合法標籤。若無法匹配，則記為 `無法判斷`，計入 invalid count。

---

## 3.6 評估指標

採用以下標準分類指標：

- **Accuracy**：整體準確率。
- **Macro Precision / Recall / F1**：三類別平均（不加權），適用於不平衡資料集。
- **Per-class Precision / Recall / F1**：分別觀察各類別表現。
- **Confusion Matrix**：揭示分類錯誤模式。
- **Invalid count**：LLM 無法產生合法標籤之筆數。

主要排序依據為 **Macro F1**，因其對不平衡類別給予公平權重。

---

# 第四章　實驗結果

## 4.1 RoBERTa 訓練過程

### 4.1.1 訓練曲線（seed=42 為例）
完整訓練在 NVIDIA RTX 4090 上耗時約 1.5 分鐘。以 seed=42 之代表性訓練為例，早停於 epoch 8（自 epoch 5 最佳值起 patience=3），各 epoch val 表現如下表：

| Epoch | Eval Loss | Accuracy | 待加強 F1 | 尚可 F1 | 精熟 F1 | Macro F1 |
|---|---|---|---|---|---|---|
| 1 | 1.032 | 0.704 | 0.000 | 0.826 | 0.000 | 0.275 |
| 2 | 0.865 | 0.761 | 0.750 | 0.851 | 0.400 | 0.667 |
| 3 | 0.380 | 0.915 | 0.857 | 0.952 | 0.783 | 0.864 |
| 4 | 0.293 | 0.958 | 0.857 | 0.980 | 0.923 | 0.920 |
| **5** ⭐ | **0.196** | **0.972** | **0.857** | **0.990** | **0.963** | **0.937** |
| 6 | 0.439 | 0.930 | 0.857 | 0.962 | 0.833 | 0.884 |
| 7 | 0.373 | 0.958 | 0.857 | 0.980 | 0.923 | 0.920 |
| 8 | 0.283 | 0.958 | 0.857 | 0.980 | 0.929 | 0.922 |

最佳 checkpoint 為 epoch 5，val macro-F1 達 0.937。`load_best_model_at_end=True` 將該 checkpoint 還原為最終模型。其他 seed 訓練曲線型態相近（前 3 個 epoch 快速收斂、第 5 epoch 附近達 val 最佳、之後略有 overfitting 跡象），故以 seed=42 為代表呈現。

### 4.1.2 最終模型於驗證集表現（seed=42）

| 指標 | 值 |
|---|---|
| Accuracy | 0.972 |
| Macro F1 | 0.937 |
| 待加強 P / R / F1 | 0.857 / 0.857 / 0.857 |
| 尚可 P / R / F1 | 1.000 / 0.980 / 0.990 |
| 精熟 P / R / F1 | 0.929 / 1.000 / 0.963 |

### 4.1.3 Multi-Seed Test Set 結果（n=5）

五個 random seed 訓練後對 test set 之表現如下：

| Seed | Test Accuracy | Macro F1 | 錯誤筆數 |
|---|---|---|---|
| 42 | 0.986 | 0.971 | 1 |
| 7 | 1.000 | 1.000 | 0 |
| 123 | 0.986 | 0.971 | 1 |
| 2024 | 0.986 | 0.971 | 1 |
| 999 | 1.000 | 1.000 | 0 |

**彙整指標 (mean ± std, n=5)**：

| 指標 | Mean ± Std | Min | Max |
|---|---|---|---|
| **Accuracy** | **0.992 ± 0.008** | 0.986 | 1.000 |
| **Macro F1** | **0.983 ± 0.016** | 0.971 | 1.000 |
| Macro Precision | 0.996 ± 0.004 | 0.994 | 1.000 |
| Macro Recall | 0.971 ± 0.026 | 0.952 | 1.000 |
| 待加強 F1 | 0.954 ± 0.042 | 0.923 | 1.000 |
| 尚可 F1 | 0.994 ± 0.005 | 0.990 | 1.000 |
| 精熟 F1 | **1.000 ± 0.000** | 1.000 | 1.000 |

值得注意的是「精熟」類別在所有 5 個 seed 下均達完美分類（F1 = 1.000，方差為零），錯誤完全集中在「待加強」與「尚可」之間的判定。詳細錯誤分析見 §4.3。

---

## 4.2 Test set 模型比較

所有模型於 72 筆 held-out test set 上之表現，依 Macro F1 由高至低排序如下。RoBERTa 為 5 個 seed 之 mean ± std，LLM 皆為 zero-shot 單次運行：

| 排名 | 模型 | Accuracy | Macro F1 | 待加強 F1 | 尚可 F1 | 精熟 F1 | Invalid |
|---|---|---|---|---|---|---|---|
| 1 | **RoBERTa (fine-tuned, n=5)** | **0.992 ± 0.008** | **0.983 ± 0.016** | 0.954 ± 0.042 | 0.994 ± 0.005 | 1.000 ± 0.000 | 0 |
| 2 | Gemini 3.1 Pro Preview | 0.889 | 0.871 | 0.933 | 0.920 | 0.759 | 0 |
| 3 | Gemini 2.5 Pro | 0.819 | 0.808 | 0.933 | 0.866 | 0.625 | 0 |
| 4 | GPT-4o-mini | 0.861 | 0.799 | 0.824 | 0.906 | 0.667 | 0 |
| 4 | Qwen3 32B | 0.861 | 0.799 | 0.824 | 0.906 | 0.667 | 0 |
| 6 | Claude Sonnet 4.5 | 0.847 | 0.796 | 0.778 | 0.891 | 0.720 | 0 |
| 7 | Gemini 3.1 Flash-Lite | 0.778 | 0.795 | 1.000 | 0.830 | 0.556 | 0 |
| 8 | Gemini 3 Flash | 0.764 | 0.792 | 1.000 | 0.813 | 0.564 | 0 |
| 9 | GPT-4.1 | 0.847 | 0.781 | 0.875 | 0.897 | 0.571 | 0 |
| 10 | o4-mini | 0.809 | 0.766 | 0.875 | 0.863 | 0.560 | 4 |
| 11 | Llama 3.3 70B | 0.806 | 0.710 | 0.737 | 0.868 | 0.526 | 0 |
| 12 | GPT-4o | 0.722 | 0.678 | 0.500 | 0.783 | 0.750 | 0 |
| 13 | Gemini 2.5 Flash-Lite | 0.708 | 0.599 | 0.583 | 0.792 | 0.421 | 0 |
| 14 | Gemini 2.5 Flash | 0.625 | 0.570 | 0.467 | 0.704 | 0.538 | 0 |
| 15 | Gemma 3 12B | 0.694 | 0.541 | 0.583 | 0.788 | 0.250 | 0 |
| 16 | Gemma 3 27B | 0.515 | 0.518 | 0.359 | 0.560 | 0.636 | 4 |
| 17 | Llama 3.1 8B | 0.194 | 0.145 | 0.241 | 0.194 | 0.000 | 0 |
| 18 | Gemma 3 4B | 0.097 | 0.068 | 0.203 | 0.000 | 0.000 | 0 |

### 4.2.1 群組比較

| 模型群組 | 平均 Macro F1 | 範圍 |
|---|---|---|
| Fine-tuned RoBERTa (n=5 seeds) | 0.983 ± 0.016 | 0.971 – 1.000 |
| 商業 LLM（GPT / Gemini / Claude） | 0.749 | 0.570 – 0.871 |
| Open-weight LLM（Llama / Qwen / Gemma） | 0.382 | 0.068 – 0.799 |

---

## 4.3 各類別錯誤模式

### 4.3.1 LLM 共通錯誤趨勢
- **「精熟」最難分**：所有 LLM 在精熟類別 F1 普遍低於待加強與尚可（多落於 0.42–0.76）。
- **「尚可」最易分**：因樣本數最多（test 中 51/72），即使弱模型仍可達 0.7+。
- **「待加強」表現分歧大**：頂尖 LLM 可達 1.000，但部分模型（如 Gemma 3 4B、Llama 3.1 8B）幾乎完全錯分。

### 4.3.2 Invalid Response 來源
- **o4-mini**：4 筆 — 推論模型在 reasoning effort 設為 low 時，token budget 200 偶有耗盡狀況，回傳空訊息。
- **Gemma 3 27B**：4 筆 — 模型回應未嚴格遵循「只回三選一」之指令，輸出長段分析文字未含目標標籤。

兩者皆對其總體 F1 造成些許下拉。

### 4.3.3 RoBERTa 邊界樣本誤分析（multi-seed）

Multi-seed 實驗中，5 個 seed 之錯誤完全集中於同一筆 test sample（idx=8），其詳細資訊如下：

| 屬性 | 值 |
|---|---|
| `Mastery_Label` (真實) | 待加強 |
| `Mastery_Ratio` | 0.167 |
| 章節 | 深度學習-遞歸神經網路(RNN) |
| 知識點 | 序列資料的處理與特徵工程 |
| 含題數 | 3 道簡答題 |
| Per-item 評分 | Incorrect、Incorrect、Partially Correct |

**各 seed 對 idx=8 的預測**：

| Seed | 預測 | 是否正確 |
|---|---|---|
| 42 | 尚可 | ✗ |
| 7 | 待加強 | ✓ |
| 123 | 尚可 | ✗ |
| 2024 | 尚可 | ✗ |
| 999 | 待加強 | ✓ |

此樣本之 `Mastery_Ratio = 0.167` 落在「待加強」類別之上界附近，屬於分類**邊界樣本**。模型在不同 random seed 下對其判定不穩定，反映 fine-tuned 模型在類別邊界區域對隨機初始化具一定敏感性，是 multi-seed 評估能揭示而單次訓練易忽略之現象。

---

# 第五章　討論

## 5.1 RoBERTa 達成近完美 (near-perfect) 準確率之成因分析

本研究 fine-tuned RoBERTa 在 test set 上達到 Macro-F1 = 0.983 ± 0.016（n=5 seeds），接近完美分類；其中兩個 seed 達到 1.000，三個 seed 各誤分一筆共同的邊界樣本。為避免讀者將此結果誤解為「模型完全勝任語意推理」，本節分層解釋其成因：

### 5.1.1 任務本質：結構化彙整 (Aggregation) 而非語意推理 (Reasoning)
`Short_Answer_Log` 中對每一道簡答題均含 per-item 評分標註（Correct / Partially Correct / Incorrect）。此 per-item 評分與整體 `Mastery_Label` 之間具強統計關聯，模型主要學習的並非「閱讀題目並判斷學生答案的正確性」，而是「將多筆 per-item 評分整合為單一掌握度等級」。換言之，模型完成的是**結構化彙整任務**。

此類任務對 RoBERTa 而言屬於相對簡單的訊號擷取與計數類問題，其 Transformer 自注意力機制能有效捕捉 log 中重複出現的關鍵詞模式。

### 5.1.2 資料集規模與分佈
Test set 規模僅 72 筆，且採 stratified split 確保類別分佈與訓練集一致。72 筆樣本對於已掌握彙整規則之模型而言並非具挑戰性之 benchmark。

### 5.1.3 訓練穩定性與邊界敏感性
Multi-seed 評估顯示，模型在絕大多數樣本上一致正確；錯誤集中於單一邊界樣本（§4.3.3, idx=8, Mastery_Ratio = 0.167）。此樣本之 per-item 評分組合（2 Incorrect + 1 Partially Correct）使其位於「待加強 / 尚可」分類閾值附近，不同 seed 訓練之模型對此細微邊界決策不一致，這是深度學習模型於類別邊界常見之現象，亦量化了本任務之訓練穩定性（macro-F1 std = 0.016）。

### 5.1.4 LLM 表現對照下的解讀
即使輸入相同，LLM 在 zero-shot 條件下並未達到相近的高準確率（最佳僅達 0.871）。這凸顯：
- **Fine-tuned 小模型**透過參數更新明確學會輸入到輸出的映射規則。
- **LLM** 受 prompt 引導傾向「閱讀題目與學生答案進行語意比對」，反而忽略 log 中明確的 per-item 評分訊號，此為任務設計與模型行為之有趣不對稱。

---

## 5.2 LLM 比較結果之觀察

### 5.2.1 模型規模與表現呈非單調關係
本實驗結果顯示，模型參數量與表現並非嚴格正相關：

- **Gemini 2.5 Flash (商業) 0.570 vs Gemini 3.1 Flash-Lite (商業) 0.795**：版本與型態差異影響大於規模。
- **Gemma 3 4B (0.068) 與 Gemma 3 27B (0.518)**：同系列規模差距大但兩者皆顯著低於商業 LLM。
- **GPT-4o (0.678) vs GPT-4o-mini (0.799)**：mini 反勝 full 版，可能因 mini 對短指令更聚焦。

此結果提醒讀者，**不能僅以模型規模推估其在特定 NLP 任務上的能力**，特別是中文教育情境下的細粒度分類任務。

### 5.2.2 商業 vs Open-weight 模型差距明顯
頂尖商業 LLM（Gemini 3.1 Pro、Gemini 2.5 Pro、GPT-4o-mini、Claude Sonnet 4.5）皆達 macro-F1 ≥ 0.79；而 open-weight 系列除 Qwen3 32B 外（macro-F1 = 0.799）皆顯著落後。Gemma 3 系列（4B/12B/27B）特別在中文 instruction 遵循上呈現弱點，Gemma 3 4B 幾乎完全失效。

### 5.2.3 Reasoning 模型未顯現明確優勢
o4-mini（reasoning_effort=low）macro-F1 為 0.766，落於商業 LLM 中段。其 4 筆 invalid response 反映 reasoning token budget 在低設定下的不穩定性。本任務複雜度可能不足以彰顯 reasoning 模型之長處。

---

## 5.3 對教育實務的啟示

### 5.3.1 自動化掌握度判定的可行性
若教學系統後台已可產生 per-item 評分（例如自動批改或老師快速標註），則 fine-tuned 模型可高精度自動產生整體掌握度等級，**減輕老師手動彙整的負擔**。本研究展示 RoBERTa 在此 pipeline 中的可用性。

### 5.3.2 LLM 作為備援方案
對於資源有限、無法 fine-tune 自己模型之教學單位，使用商業 LLM 直接完成此任務亦可達 macro-F1 0.8+，雖低於 fine-tuned 模型，但仍具實用價值。

---

## 5.4 限制與未來工作

### 5.4.1 限制 (Limitations)
1. **資料集規模有限**：474 筆，test set 72 筆，難以代表更廣泛場景。
2. **領域特定**：所有資料皆來自特定課程之知識點，跨領域泛化能力未測試。
3. **Input 含監督訊號**：`Short_Answer_Log` 中之 per-item 評分對最終 label 提供強訊號，模型實際學習的更接近結構化彙整而非純語意推理。
4. **LLM 為單次運行**：商業 / open-weight LLM 雖設定 `temperature=0`，仍可能有 server-side 微小 jitter，未進行 multi-run 評估。

### 5.4.2 未來工作 (Future Work)
1. **擴大資料集**：蒐集更多學生與更多領域之資料，提升模型泛化能力。
2. **去除 per-item 評分之 ablation 實驗**：評估純從學生作答內容推測掌握度之模型表現，作為更嚴格之語意推理測試。
3. **Prompt 工程強化**：針對 LLM 設計 chain-of-thought 或 few-shot prompt，比較其與 fine-tuned 模型之差距。
4. **錯誤回應處理**：對 o4-mini、Gemma 27B 之 invalid response 引入 self-consistency 投票或重試機制，提升公平性。
5. **邊界樣本之強化處理**：對如 idx=8 之 borderline samples 引入 ordinal regression 或 label smoothing，緩解閾值附近的判定不穩定。
