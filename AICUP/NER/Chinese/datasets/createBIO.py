from collections import defaultdict
import json
from transformers.models.whisper.english_normalizer import BasicTextNormalizer, EnglishTextNormalizer


english_normalizer = BasicTextNormalizer()

def normalize_text(text_to_normalize):
    if english_normalizer:
        return english_normalizer(text_to_normalize)
    return text_to_normalize

def convert_to_bio_json(transcription_file, label_file, output_file=None):
    """
    將兩個txt檔案直接轉換成JSON格式的BIO標註
    
    Args:
        transcription_file: 第一個檔案路徑 (格式: filename\ttranscription)
        label_file: 第二個檔案路徑 (格式: filename\tlabel\tstartTime\tendTime\ttext)
        output_file: 輸出JSON檔案路徑 (可選)
    
    Returns:
        dict: filename -> {"tokens": [...], "labels": [...]} 的BIO格式JSON
    """
    
    # 1. 讀取transcription檔案
    transcription_dict = {}
    print("讀取transcription檔案...")
    
    with open(transcription_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) < 2:
                print(f"警告: 第{line_num}行格式錯誤，跳過")
                continue
                
            filename = parts[0]
            transcription = '\t'.join(parts[1:])
            transcription = normalize_text(transcription)
            words = list(transcription)
            transcription_dict[filename] = words
    
    print(f"成功讀取 {len(transcription_dict)} 個transcription")
    
    # 2. 讀取label檔案
    label_dict = defaultdict(list)
    print("讀取label檔案...")
    
    with open(label_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split('\t')
            if len(parts) < 5:
                print(f"警告: 第{line_num}行格式錯誤，跳過")
                continue
                
            filename = parts[0]
            label = parts[1]
            start_time = float(parts[2])
            end_time = float(parts[3])
            text = '\t'.join(parts[4:])
            text = normalize_text(text)
            
            label_dict[filename].append((label, start_time, end_time, text))
    
    print(f"成功讀取 {sum(len(labels) for labels in label_dict.values())} 個標註")
    
    # 3. 對標註依startTime排序
    for filename in label_dict:
        label_dict[filename].sort(key=lambda x: x[1])
    
    # 4. 轉換成BIO格式JSON
    bio_json = {}
    
    for filename, words in transcription_dict.items():
        bio_labels = ['O'] * len(words)
        
        if filename not in label_dict:
            bio_json[filename] = {"tokens": words, "labels": bio_labels}
            continue
        
        labels_for_file = label_dict[filename]
        
        for label, start_time, end_time, text in labels_for_file:
            label_words = list(text)
            matched = False
            
            # 在transcription中尋找匹配的詞序列
            for i in range(len(words) - len(label_words) + 1):
                if words[i:i+len(label_words)] == label_words:
                    if len(label_words) > 0:
                        bio_labels[i] = f'B-{label}'
                        for j in range(1, len(label_words)):
                            bio_labels[i + j] = f'I-{label}'
                    matched = True
                    break
            
            if not matched:
                print(f"警告: 在 {filename} 中找不到匹配的文本: '{text}'")
        
        bio_json[filename] = {"tokens": words, "labels": bio_labels}
    
    # 5. 保存JSON檔案
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(bio_json, f, ensure_ascii=False, indent=2)
        print(f"JSON檔案已保存到: {output_file}")
    
    return bio_json

def convert_to_training_format(bio_json, output_file=None):
    """
    將BIO JSON轉換成訓練用的格式 (每個檔案變成一個訓練樣本)
    
    Args:
        bio_json: convert_to_bio_json的輸出
        output_file: 輸出檔案路徑
    
    Returns:
        list: 訓練用的數據格式
    """
    training_data = []
    
    for i, (filename, data) in enumerate(bio_json.items()):
        training_sample = {
            "id": i,
            "tokens": data["tokens"],
            "labels": data["labels"]
        }
        training_data.append(training_sample)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        print(f"訓練格式檔案已保存到: {output_file}")
    
    return training_data

def print_json_preview(bio_json, max_files=3):
    """顯示JSON結果預覽"""
    print("\n" + "="*50)
    print("BIO JSON格式預覽:")
    print("="*50)
    
    for i, (filename, data) in enumerate(bio_json.items()):
        if i >= max_files:
            print(f"... 還有 {len(bio_json) - max_files} 個檔案")
            break
        
        print(f"\n檔案: {filename}")
        print("-" * 30)
        print("Tokens:", data["tokens"][:10], "..." if len(data["tokens"]) > 10 else "")
        print("Labels:", data["labels"][:10], "..." if len(data["labels"]) > 10 else "")
        
        # 顯示實體統計
        entities = []
        i = 0
        while i < len(data["labels"]):
            if data["labels"][i].startswith('B-'):
                entity_type = data["labels"][i][2:]
                entity_words = [data["tokens"][i]]
                i += 1
                while i < len(data["labels"]) and data["labels"][i] == f'I-{entity_type}':
                    entity_words.append(data["tokens"][i])
                    i += 1
                entities.append((entity_type, ' '.join(entity_words)))
            else:
                i += 1
        
        if entities:
            print("實體:")
            for entity_type, entity_text in entities[:5]:
                print(f"  - {entity_type}: {entity_text}")
# ================================
# 主要執行代碼
# ================================

if __name__ == "__main__":
    
    # 轉換為BIO JSON格式
    bio_json = convert_to_bio_json(
        r'C:\Users\C110151154\PycharmProjects\NeMo\AICUP\datasets\chinese\task1_answer.txt',
        r'C:\Users\C110151154\PycharmProjects\NeMo\AICUP\datasets\chinese\task2_answer.txt',
        './chinese_bio_raw.json'
    )
    
    # 顯示預覽
    print_json_preview(bio_json)
    
    # 轉換為訓練格式
    training_data = convert_to_training_format(
        bio_json, 
        './chinese_bio.json'
    )
    
    print(f"\n轉換完成！")
    print(f"- BIO JSON: bio_output.json")
    print(f"- 訓練格式: training_data.json")
    print(f"- 總共 {len(bio_json)} 個檔案")
    print(f"- 總共 {len(training_data)} 個訓練樣本")
