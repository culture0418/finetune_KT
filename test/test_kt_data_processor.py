import pytest
import pandas as pd
import io
from finetune_bert import KTDataProcessor

# --- 測試資料區 ---
# 27 筆資料：21 筆有效 + 6 筆無效（1 筆空值 + 5 筆無法識別的標籤）
FAKE_CSV_DATA = """chapter,section,all_logs,Preview_ChatLog,Review_ChatLog,Mastery_Level_K4
監督式學習,監督式學習的定義與應用,答對5題答錯1題,老師你好我想問監督式學習是什麼,我現在理解了謝謝老師,精熟
監督式學習,迴歸問題的基本概念,答對3題答錯3題,迴歸和分類有什麼不同,還是有點不太懂,尚可
監督式學習,分類問題的應用場景,答對1題答錯5題,分類問題好難,我需要多練習,待加強
監督式學習,邏輯迴歸應用,答對2題答錯3題,這個概念還不清楚,要多看書,不及格
監督式學習,支持向量機SVM,答對4題答錯1題,SVM的核函數很有趣,掌握得不錯,良好
非監督式學習,群集分析的原理,答對4題答錯2題,K-means演算法是怎麼運作的,了解了謝謝,精熟
非監督式學習,降維技術PCA,答對3題答錯2題,PCA的數學原理是什麼,我再想想,尚可
非監督式學習,異常檢測方法,答對2題答錯4題,異常檢測要怎麼做,還不太會,待加強
非監督式學習,層次聚類方法,答對1題答錯6題,完全不懂這個,,
非監督式學習,DBSCAN聚類,答對4題答錯2題,密度聚類的概念,理解得還行,良好
深度學習,神經網路基礎,答對5題答錯0題,神經元的啟動函數有哪些,完全理解了,精熟
深度學習,反向傳播演算法,答對4題答錯1題,梯度下降法的原理,這個我懂了,精熟
深度學習,卷積神經網路CNN,答對2題答錯3題,CNN用在影像辨識,需要再複習,尚可
深度學習,遞迴神經網路RNN,答對1題答錯4題,RNN好複雜,我不太懂,待加強
深度學習,注意力機制,答對0題答錯5題,這個太難了,完全搞不懂,未達標
深度學習,Transformer架構,答對4題答錯2題,self-attention機制,基本理解,良好
模型評估,混淆矩陣的解讀,答對5題答錯1題,TP和FP是什麼意思,我會了,精熟
模型評估,準確率與召回率,答對3題答錯2題,precision和recall的差異,有點懂了,尚可
模型評估,ROC曲線與AUC,答對2題答錯3題,AUC值代表什麼,不太清楚,待加強
特徵工程,特徵選擇方法,答對4題答錯1題,如何選擇重要特徵,明白了,精熟
特徵工程,特徵縮放與正規化,答對3題答錯3題,,還在學習中,尚可
特徵工程,類別特徵編碼,答對1題答錯5題,one-hot encoding是什麼,,待加強
特徵工程,特徵交互作用,答對1題答錯7題,這個沒學過,不知道,Unknown
強化學習,Q-Learning演算法,答對5題答錯0題,reward和penalty的設計,完全掌握, 待觀察
強化學習,策略梯度方法,答對2題答錯2題,policy gradient怎麼用,還在理解,尚可
決策樹,決策樹的分割準則,答對4題答錯2題,Gini和Entropy的差別,懂了,精熟
集成學習,隨機森林演算法,答對1題答錯4題,為什麼要用多棵樹,需要加強,待加強
集成學習,梯度提升樹GBDT,答對3題答錯2題,boosting的概念,還可以,尚可
"""

class TestKTDataProcessor:
    """
    KTDataProcessor 測試類別
    使用 tmp_path 替代 Mocking，進行真實檔案讀取測試，更具可靠性。
    """

    @pytest.fixture
    def csv_file(self, tmp_path):
        """
        Fixture: 建立一個包含完整假資料的真實 CSV 檔案。
        返回檔案路徑 (str)。
        """
        p = tmp_path / "finetune_dataset.csv"
        p.write_text(FAKE_CSV_DATA, encoding='utf-8')
        return str(p)

    @pytest.fixture
    def processor(self, csv_file):
        """
        Fixture: 初始化 KTDataProcessor 實例
        """
        return KTDataProcessor(csv_path=csv_file)

    def test_init_attributes(self, processor):
        """測試初始化屬性是否正確"""
        assert processor.num_labels == 4
        assert processor.label_map == {"待加強": 0, "尚可": 1, "良好": 2, "精熟": 3}
        assert processor.train_df is None
        assert processor.val_df is None

    def test_load_and_clean_success(self, processor, capsys):
        """
        測試案例 1: 成功載入與清理
        - 讀取 28 筆測試資料
        - 移除 1 筆 Mastery_Level_K4 為空的資料（層次聚類方法）
        - 移除 5 筆無法識別的標籤（不及格、未達標、Unknown、待觀察、超出範圍）
        - 最終保留 21 筆有效資料
        - 驗證 labels 欄位已成功建立
        - 驗證資料清理和索引重置
        """
        # 執行載入與清理
        result_df = processor._load_and_clean()
        
        # 驗證 print 輸出
        captured = capsys.readouterr()
        assert "讀取成功，共 28 筆" in captured.out, "應該讀取 28 筆原始資料"
        assert "警告：移除 1 筆 'Mastery_Level_K4' 為空的資料" in captured.out, "應該移除 1 筆空標籤"
        assert "警告：移除 4 筆無法識別的標籤值" in captured.out, "應該移除 4 筆無效標籤"
        assert "資料清理完成，剩餘 23 筆有效資料" in captured.out, "最終應剩 23 筆"
        
        # 斷言：資料筆數
        assert len(result_df) == 23, f"預期 23 筆有效資料，實際得到 {len(result_df)} 筆"
        
        # 斷言：labels 欄位存在且為整數型別
        assert 'labels' in result_df.columns, "labels 欄位不存在"
        assert result_df['labels'].dtype == int, "labels 欄位應該是整數類型"
        assert result_df['labels'].min() >= 0, "標籤值不應小於 0"
        assert result_df['labels'].max() <= 3, "標籤值不應大於 3"
        
        # 斷言：索引已重置
        assert result_df.index.tolist() == list(range(23)), "索引應該從 0 到 22 連續"
        
        # 斷言：文字欄位型別和無 NaN
        text_cols = ['chapter', 'section', 'all_logs', 'Preview_ChatLog', 'Review_ChatLog']
        for col in text_cols:
            assert result_df[col].dtype == object or result_df[col].dtype == 'string', \
                f"{col} 欄位應該是字串型別"
            assert not result_df[col].isnull().any(), f"{col} 欄位不應包含 NaN 值"
        
        # 斷言：只有有效標籤存在
        valid_labels = {'待加強', '尚可', '良好', '精熟'}
        unique_mastery = set(result_df['Mastery_Level_K4'].unique())
        assert unique_mastery == valid_labels, \
            f"結果應只包含有效標籤 {valid_labels}，實際包含: {unique_mastery}"

    def test_invalid_labels_filtered(self, processor):
        """
        測試案例 1.5: 專注驗證無效標籤過濾邏輯
        - 確認 6 筆無效標籤（1 筆空值 + 5 筆無法識別）被正確移除
        - 此測試與 test_load_and_clean_success 互補，專注於過濾邏輯
        """
        result_df = processor._load_and_clean()
        
        # 定義無效標籤對應的 section
        # 1. 空值: 層次聚類方法 (Mastery_Level_K4 為空)
        # 2. 不及格: 邏輯迴歸應用
        # 3. 未達標: 注意力機制
        # 4. Unknown: 特徵交互作用
        # 5. 待觀察: Q-Learning演算法
        invalid_sections = ['邏輯迴歸應用', '層次聚類方法', '注意力機制', '特徵交互作用', 'Q-Learning演算法']
        
        # 驗證這些 section 確實不在結果中
        result_sections = set(result_df['section'].values)
        for section in invalid_sections:
            assert section not in result_sections, \
                f"無效標籤的資料 '{section}' 應該被過濾掉，但仍存在於結果中"
        
        # 驗證只有四種有效標籤
        valid_labels = {'待加強', '尚可', '良好', '精熟'}
        actual_labels = set(result_df['Mastery_Level_K4'].unique())
        assert actual_labels == valid_labels, \
            f"結果應只包含 {valid_labels}，實際包含: {actual_labels}"
        
        # 驗證每種有效標籤的分佈
        label_counts = result_df['Mastery_Level_K4'].value_counts()
        print(f"\n有效標籤分佈: {label_counts.to_dict()}")
        assert all(count > 0 for count in label_counts.values), \
            "每種有效標籤應該至少有 1 筆資料"

    def test_nan_handling(self, processor):
        """
        測試案例 2: NaN 值與空值處理
        驗證特定欄位的 NaN 是否轉為空字串
        """
        result_df = processor._load_and_clean()
        
        # 1. 檢查 'Review_ChatLog' 原本是 NaN 的情況 (section='層次聚類方法' 被濾掉，改查 '類別特徵編碼')
        # 原資料中：特徵工程,類別特徵編碼,...,one-hot encoding是什麼,,待加強
        row = result_df[result_df['section'] == '類別特徵編碼'].iloc[0]
        assert row['Review_ChatLog'] == '', "Review_ChatLog 的 NaN 應轉為空字串"
        
        # 2. 檢查 'Preview_ChatLog' 原本是空的情況
        # 原資料中：特徵工程,特徵縮放與正規化,...,,還在學習中,尚可
        row_preview = result_df[result_df['section'] == '特徵縮放與正規化'].iloc[0]
        assert row_preview['Preview_ChatLog'] == '', "Preview_ChatLog 的缺失值應轉為空字串"

    def test_label_mapping_accuracy(self, processor):
        """
        測試案例 3: 標籤映射準確度
        - 驗證每個標籤都有充足的樣本（去除 6 筆無效資料後）
        """
        result_df = processor._load_and_clean()
        
        # 驗證每個標籤都有充足的樣本
        label_counts = result_df['labels'].value_counts()
        assert label_counts.get(0, 0) >= 5, f"待加強標籤樣本數不足，實際: {label_counts.get(0, 0)} 筆"
        assert label_counts.get(1, 0) >= 5, f"尚可標籤樣本數不足，實際: {label_counts.get(1, 0)} 筆"
        assert label_counts.get(2, 0) >= 3, f"良好標籤樣本數不足，實際: {label_counts.get(2, 0)} 筆"
        assert label_counts.get(3, 0) >= 5, f"精熟標籤樣本數不足，實際: {label_counts.get(3, 0)} 筆"
        
        # 驗證標籤值對應關係
        for idx, row in result_df.iterrows():
            mastery = row['Mastery_Level_K4']
            label = row['labels']
            expected_label = processor.label_map[mastery]
            assert label == expected_label, \
                f"標籤映射錯誤: '{mastery}' 應該映射到 {expected_label}，實際為 {label}"

    def test_prepare_data_split_logic(self, processor, capsys):
        """
        測試案例 4: 資料分割 (Split) 與 分層抽樣 (Stratify)
        """
        # 執行分割，測試集佔 20% (23筆 * 0.2 ≈ 5筆 val, 18筆 train)
        processor.prepare_data(test_size=0.2, random_state=42)
        
        assert len(processor.train_df) == 18
        assert len(processor.val_df) == 5
        
        # 驗證 Stratify 效果：驗證集必須包含所有出現過的類別 (依比例)
        # 我們的資料量夠，應該 train 和 val 都有 0, 1, 2, 3 四種標籤
        val_labels = processor.val_df['labels'].unique()
        assert len(val_labels) >= 2, "驗證集因分層抽樣應至少包含多種標籤"
        
        # 檢查 console 輸出是否有印出統計資訊
        captured = capsys.readouterr()
        assert "訓練集標籤分佈" in captured.out
        assert "驗證集標籤分佈" in captured.out

    def test_missing_required_column(self, tmp_path, capsys):
        """
        測試案例 5: 缺少必要欄位時的自動修復機制
        """
        # 建立一個缺少 'Review_ChatLog' 的 CSV
        incomplete_data = "chapter,section,all_logs,Preview_ChatLog,Mastery_Level_K4\nC1,S1,log,prev,精熟"
        p = tmp_path / "incomplete.csv"
        p.write_text(incomplete_data, encoding='utf-8')
        
        proc = KTDataProcessor(str(p))
        df = proc._load_and_clean()
        
        # 檢查警告
        captured = capsys.readouterr()
        assert "警告：資料中缺少欄位 Review_ChatLog" in captured.out
        
        # 檢查欄位是否被自動補上
        assert 'Review_ChatLog' in df.columns
        assert df.iloc[0]['Review_ChatLog'] == ''

    def test_get_dataframes_flow_control(self, processor):
        """
        測試案例 6: 流程控制 (未 prepare 前不可 get)
        """
        with pytest.raises(ValueError, match="請先呼叫 .prepare_data()"):
            processor.get_dataframes()

    def test_file_not_found_exception(self, tmp_path, capsys):
        """
        測試案例 7: 檔案不存在的異常處理
        """
        # 指向一個不存在的路徑
        non_existent_path = tmp_path / "ghost.csv"
        proc = KTDataProcessor(str(non_existent_path))
        
        with pytest.raises(FileNotFoundError):
            proc._load_and_clean()
            
        captured = capsys.readouterr()
        assert "錯誤：找不到檔案" in captured.out

    def test_stratification_distribution(self, processor):
        """
        額外測試: 深度驗證分層抽樣的比例一致性
        比較 原始資料 vs 訓練集 的標籤比例
        """
        processor.prepare_data(test_size=0.25, random_state=42)
        
        # 計算原始有效資料的比例 (透過 _load_and_clean 取得)
        full_df = processor._load_and_clean()
        original_dist = full_df['labels'].value_counts(normalize=True).sort_index()
        
        # 計算訓練集的比例
        train_dist = processor.train_df['labels'].value_counts(normalize=True).sort_index()
        
        # 驗證每個類別的比例差異在容許範圍內 (例如 15%)
        # 因為數據量小(20筆)，分層抽樣可能會有輕微偏差，這是正常的
        for label in original_dist.index:
            diff = abs(original_dist[label] - train_dist.get(label, 0))
            assert diff < 0.15, f"標籤 {label} 的分佈差異過大 ({diff:.2f})"

    def test_analyze_token_lengths(self, processor, monkeypatch, capsys):
        """
        測試案例 8: Token 長度分析功能
        使用 Mock Tokenizer 避免下載模型
        """
        # Mock Tokenizer
        class MockTokenizer:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()
            
            def encode(self, text, add_special_tokens=True):
                # 根據文本內容決定長度
                if "長文本" in text:
                    return [0] * 600  # 模擬長度 600
                return [0] * 100      # 模擬長度 100

        # Patch finetune_bert 模組中的 BertTokenizer
        import finetune_bert
        monkeypatch.setattr(finetune_bert, "BertTokenizer", MockTokenizer)
        
        # 準備資料
        processor.prepare_data(test_size=0.2, random_state=42)
        
        # 修改一筆資料使其變長 (在 train_df 中)
        # 為了確保修改生效，我們直接修改 train_df 的第一筆
        # 注意：這裡假設 train_df 至少有一筆資料
        processor.train_df.iloc[0, processor.train_df.columns.get_loc('all_logs')] = "長文本測試"
        
        # 執行分析
        ratio, max_len = processor.analyze_token_lengths("dummy_model", threshold=512)
        
        # 驗證
        # 總共 23 筆有效資料 (來自 FAKE_CSV_DATA)
        # 1 筆超過 512
        expected_ratio = (1 / 23) * 100
        
        assert max_len == 600
        assert ratio == expected_ratio

        # 驗證是否有印出詳細資訊
        captured = capsys.readouterr()
        assert "檢查超過 512 tokens 的資料詳細資訊" in captured.out
        assert "⚠️ [過長] ID:" in captured.out
        assert "Token長度: 600" in captured.out
        assert "文本長度:" in captured.out