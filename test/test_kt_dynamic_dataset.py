import pytest
import pandas as pd
import torch
from transformers import BertTokenizer
from finetune_bert import KTDynamicDataset


class TestKTDynamicDataset:
    """
    測試 KTDynamicDataset 類別的所有功能
    """
    
    @pytest.fixture
    def sample_dataframe(self):
        """建立測試用的 DataFrame"""
        return pd.DataFrame({
            'chapter': ['監督式學習', '非監督式學習', '深度學習', '機器學習'],
            'section': ['線性迴歸', 'K-means聚類', '神經網路基礎', '決策樹'],
            'all_logs': ['答對5題答錯1題', '答對3題答錯3題', '答對4題答錯2題', '答對2題答錯4題'],
            'Preview_ChatLog': ['這是什麼概念', '如何使用', '原理是什麼', '決策樹怎麼用'],
            'Review_ChatLog': ['我懂了', '還需要練習', '完全理解', '需要加強'],
            'Mastery_Level_K4': ['精熟', '尚可', '良好', '待加強'],
            'labels': [3, 1, 2, 0]
        })
    
    @pytest.fixture
    def tokenizer(self):
        """載入 BERT Tokenizer（使用中文 BERT）"""
        return BertTokenizer.from_pretrained('bert-base-chinese')
    
    @pytest.fixture
    def dataset(self, sample_dataframe, tokenizer):
        """建立 Dataset 實例"""
        return KTDynamicDataset(sample_dataframe, tokenizer, max_token_len=128)
    
    def test_init_attributes(self, dataset, sample_dataframe, tokenizer):
        """測試初始化屬性是否正確"""
        assert dataset.max_len == 128
        assert len(dataset.data) == 4
        assert dataset.tokenizer == tokenizer
        assert list(dataset.data.index) == [0, 1, 2, 3], "索引應該被重置"
    
    def test_len(self, dataset):
        """測試 __len__ 方法"""
        assert len(dataset) == 4
    
    def test_getitem_return_structure(self, dataset):
        """測試 __getitem__ 返回的資料結構"""
        item = dataset[0]
        
        # 驗證返回字典包含所需的 key
        assert isinstance(item, dict)
        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert 'labels' in item
        
        # 驗證資料型別
        assert isinstance(item['input_ids'], torch.Tensor)
        assert isinstance(item['attention_mask'], torch.Tensor)
        assert isinstance(item['labels'], torch.Tensor)
    
    def test_getitem_tensor_shapes(self, dataset):
        """測試返回的 tensor 形狀"""
        item = dataset[0]
        
        # 所有 tensor 應該是 1D，長度為 max_len
        assert item['input_ids'].dim() == 1
        assert item['attention_mask'].dim() == 1
        assert item['labels'].dim() == 0  # 標籤是 scalar
        
        assert item['input_ids'].shape[0] == 128
        assert item['attention_mask'].shape[0] == 128
    
    def test_getitem_label_mapping(self, dataset, sample_dataframe):
        """測試標籤映射是否正確"""
        for idx in range(len(dataset)):
            item = dataset[idx]
            expected_label = sample_dataframe.iloc[idx]['labels']
            assert item['labels'].item() == expected_label
    
    def test_text_formatting(self, dataset, tokenizer):
        """測試文本格式化邏輯"""
        item = dataset[0]
        
        # 解碼 input_ids 來檢查文本內容
        decoded_text = tokenizer.decode(item['input_ids'], skip_special_tokens=True)
        
        # bert-base-chinese 使用字符級分詞，解碼後每個字之間會有空格
        # 例如：'監督式學習' -> '監 督 式 學 習'
        # 移除空格後再進行比對
        decoded_text_no_space = decoded_text.replace(' ', '')
        
        # 驗證關鍵字是否出現在文本中
        assert '監督式學習' in decoded_text_no_space, \
            f"期望找到 '監督式學習'，實際文本: {decoded_text_no_space[:100]}"
        assert '線性迴歸' in decoded_text_no_space, \
            f"期望找到 '線性迴歸'，實際文本: {decoded_text_no_space[:100]}"
        assert '答對5題答錯1題' in decoded_text_no_space
        assert '這是什麼概念' in decoded_text_no_space
        assert '我懂了' in decoded_text_no_space
        assert '精熟' in decoded_text_no_space
        
        # 驗證結構化關鍵詞
        assert '章節' in decoded_text_no_space or '章节' in decoded_text_no_space
        assert '知識點' in decoded_text_no_space or '知识点' in decoded_text_no_space
        assert '作答紀錄' in decoded_text_no_space or '作答纪录' in decoded_text_no_space
        assert '學生掌握度' in decoded_text_no_space or '学生掌握度' in decoded_text_no_space
    
    def test_special_tokens(self, dataset, tokenizer):
        """測試特殊 token 是否正確添加"""
        item = dataset[0]
        
        # [CLS] token 應該在開頭
        assert item['input_ids'][0] == tokenizer.cls_token_id, \
            f"第一個 token 應該是 [CLS]({tokenizer.cls_token_id})，實際為 {item['input_ids'][0]}"
        
        # 找到第一個 [SEP] token（非 padding）
        non_pad_ids = item['input_ids'][item['attention_mask'] == 1]
        assert tokenizer.sep_token_id in non_pad_ids, \
            "[SEP] token 應該出現在非 padding 區域"
    
    def test_padding(self, dataset, tokenizer):
        """測試 padding 是否正確"""
        item = dataset[0]
        
        # 找到 padding 的位置
        pad_positions = (item['input_ids'] == tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
        
        if len(pad_positions) > 0:
            # 驗證 padding 位置的 attention_mask 為 0
            for pos in pad_positions:
                assert item['attention_mask'][pos] == 0
    
    def test_attention_mask(self, dataset):
        """測試 attention mask 的正確性"""
        item = dataset[0]
        
        # attention_mask 應該只包含 0 和 1
        unique_values = torch.unique(item['attention_mask'])
        assert all(val in [0, 1] for val in unique_values.tolist())
        
        # 至少應該有一些 token 不是 padding（attention_mask = 1）
        assert (item['attention_mask'] == 1).sum() > 0
    
    def test_truncation(self, sample_dataframe, tokenizer):
        """測試超長文本的截斷功能"""
        # 建立一個很長的文本
        long_df = sample_dataframe.copy()
        long_df.loc[0, 'all_logs'] = '答對' * 1000  # 製造超長文本
        
        dataset = KTDynamicDataset(long_df, tokenizer, max_token_len=128)
        item = dataset[0]
        
        # 驗證長度確實被截斷到 max_len
        assert item['input_ids'].shape[0] == 128
        assert item['attention_mask'].shape[0] == 128
    
    def test_empty_fields(self, tokenizer):
        """測試空欄位的處理"""
        df_with_empty = pd.DataFrame({
            'chapter': ['深度學習'],
            'section': [''],  # 空欄位
            'all_logs': [''],  # 空欄位
            'Preview_ChatLog': ['有問題'],
            'Review_ChatLog': [''],  # 空欄位
            'Mastery_Level_K4': ['精熟'],
            'labels': [2]
        })
        
        dataset = KTDynamicDataset(df_with_empty, tokenizer, max_token_len=128)
        item = dataset[0]
        
        # 應該能正常處理，不拋出異常
        assert item['input_ids'].shape[0] == 128
        assert item['labels'].item() == 2
    
    def test_batch_consistency(self, dataset):
        """測試同一筆資料多次讀取的一致性"""
        item1 = dataset[1]
        item2 = dataset[1]
        
        # 同一筆資料應該產生相同的結果
        assert torch.equal(item1['input_ids'], item2['input_ids'])
        assert torch.equal(item1['attention_mask'], item2['attention_mask'])
        assert torch.equal(item1['labels'], item2['labels'])
    
    def test_different_max_lengths(self, sample_dataframe, tokenizer):
        """測試不同的 max_length 設定"""
        lengths = [64, 128, 256, 512]
        
        for max_len in lengths:
            dataset = KTDynamicDataset(sample_dataframe, tokenizer, max_token_len=max_len)
            item = dataset[0]
            
            assert item['input_ids'].shape[0] == max_len
            assert item['attention_mask'].shape[0] == max_len
    
    def test_all_labels_types(self, tokenizer):
        """測試四種標籤類型"""
        df_all_labels = pd.DataFrame({
            'chapter': ['ch1', 'ch2', 'ch3', 'ch4'],
            'section': ['s1', 's2', 's3', 's4'],
            'all_logs': ['log1', 'log2', 'log3', 'log4'],
            'Preview_ChatLog': ['pre1', 'pre2', 'pre3', 'pre4'],
            'Review_ChatLog': ['post1', 'post2', 'post3', 'post4'],
            'Mastery_Level_K4': ['待加強', '尚可', '良好', '精熟'],
            'labels': [0, 1, 2, 3]
        })
        
        dataset = KTDynamicDataset(df_all_labels, tokenizer)
        
        # 驗證四種標籤都能正確處理
        assert dataset[0]['labels'].item() == 0  # 待加強
        assert dataset[1]['labels'].item() == 1  # 尚可
        assert dataset[2]['labels'].item() == 2  # 良好
        assert dataset[3]['labels'].item() == 3  # 精熟
    
    def test_index_out_of_bounds(self, dataset):
        """測試索引超出範圍"""
        with pytest.raises(IndexError):
            _ = dataset[10]  # 只有 3 筆資料
    
    def test_negative_index(self, dataset, sample_dataframe):
        """測試負數索引"""
        # pandas iloc 支援負數索引
        item = dataset[-1]  # 最後一筆
        
        # 取得最後一筆的預期標籤
        expected_label = sample_dataframe.iloc[-1]['labels']
        assert item['labels'].item() == expected_label, \
            f"最後一筆的標籤應該是 {expected_label}，實際為 {item['labels'].item()}"
        
        # 驗證 -1 和正數索引取得的結果相同
        last_item_positive = dataset[len(dataset) - 1]
        assert torch.equal(item['input_ids'], last_item_positive['input_ids'])
        assert torch.equal(item['labels'], last_item_positive['labels'])
    
    def test_dataframe_reset_index(self, tokenizer):
        """測試 DataFrame 索引重置"""
        # 建立一個索引不連續的 DataFrame
        df = pd.DataFrame({
            'chapter': ['ch1', 'ch2', 'ch3'],
            'section': ['s1', 's2', 's3'],
            'all_logs': ['log1', 'log2', 'log3'],
            'Preview_ChatLog': ['pre1', 'pre2', 'pre3'],
            'Review_ChatLog': ['post1', 'post2', 'post3'],
            'Mastery_Level_K4': ['精熟', '尚可', '待加強'],
            'labels': [2, 1, 0]
        }, index=[5, 10, 15])  # 不連續的索引
        
        dataset = KTDynamicDataset(df, tokenizer)
        
        # 驗證內部 DataFrame 的索引已被重置
        assert list(dataset.data.index) == [0, 1, 2]
        
        # 驗證可以正常訪問
        item = dataset[0]
        assert item['labels'].item() == 2
    
    def test_tokenizer_output_format(self, dataset, tokenizer):
        """測試 tokenizer 輸出格式"""
        item = dataset[0]
        
        # 驗證 dtype
        assert item['input_ids'].dtype == torch.long
        assert item['attention_mask'].dtype == torch.long
        assert item['labels'].dtype == torch.long
        
        # 驗證值的範圍
        vocab_size = tokenizer.vocab_size
        assert (item['input_ids'] >= 0).all()
        assert (item['input_ids'] < vocab_size).all()
    
    def test_chinese_text_encoding(self, tokenizer):
        """專門測試中文文本的編碼是否正確"""
        # 測試簡單的中文句子
        test_df = pd.DataFrame({
            'chapter': ['機器學習'],
            'section': ['線性回歸'],
            'all_logs': ['答對三題'],
            'Preview_ChatLog': ['老師好'],
            'Review_ChatLog': ['謝謝'],
            'Mastery_Level_K4': ['精熟'],
            'labels': [2]
        })
        
        dataset = KTDynamicDataset(test_df, tokenizer, max_token_len=128)
        item = dataset[0]
        
        # 解碼並驗證
        decoded = tokenizer.decode(item['input_ids'], skip_special_tokens=True)
        decoded_no_space = decoded.replace(' ', '')
        
        assert '機器學習' in decoded_no_space
        assert '線性回歸' in decoded_no_space
        assert '答對三題' in decoded_no_space
        assert '老師好' in decoded_no_space
        assert '謝謝' in decoded_no_space
    
    def test_tokenizer_vocab_size(self, tokenizer):
        """驗證中文 BERT tokenizer 的詞彙表大小"""
        # bert-base-chinese 的詞彙表大小應該是 21128
        assert tokenizer.vocab_size == 21128, \
            f"bert-base-chinese 的 vocab_size 應該是 21128，實際為 {tokenizer.vocab_size}"
