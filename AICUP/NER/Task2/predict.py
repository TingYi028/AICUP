import json
import os
import numpy as np
import torch
from torch import nn
# from torch.nn import functional as F # Not strictly needed for predict by TokenClassificationWithCRF forward logic for inference
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PreTrainedModel,
    AutoModel
)
from transformers.modeling_outputs import TokenClassifierOutput
from torchcrf import CRF
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer


# Safetensors is usually handled by transformers' from_pretrained, but good to be aware if manual loading was needed.
# from safetensors.torch import load_file as load_safetensors_file # Not needed if using from_pretrained

# --- 複製 train.py 中的 TokenClassificationWithCRF 類別 ---
class TokenClassificationWithCRF(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        # AutoModel.from_pretrained 會自動載入 backbone 的權重
        self.backbone = AutoModel.from_pretrained(
            "microsoft/deberta-v3-large",
            config=config,  # Pass the full config for consistency, AutoModel filters what it needs
        )
        classifier_dropout = getattr(config, 'classifier_dropout', None)
        if classifier_dropout is None:
            classifier_dropout = getattr(config, 'hidden_dropout_prob', 0.1)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            return_dict=None,
            **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
            **kwargs
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            crf_mask = attention_mask.bool() if attention_mask is not None else torch.ones_like(
                labels, dtype=torch.bool, device=labels.device
            )
            active_labels_for_crf = labels.clone()
            # 忽略 -100 標籤，CRF 模組本身不處理 -100，所以將其設為有效標籤，但其 mask 為 False 不參與損失計算
            active_labels_for_crf[labels == -100] = 0  # 假設 0 是一個安全的填充或 "O" 標籤 ID
            loss = -self.crf(
                emissions=logits,
                tags=active_labels_for_crf,
                mask=crf_mask,
                reduction='mean'
            )
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# --- 配置 ---
MODEL_PATH = r"results_ner_microsoft/deberta-v3-large-crf/checkpoint-1048"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_JSON_PATH = r"../ASR/test_result/task1_answer_en.json"
OUTPUT_TXT_PATH = r"./test_result/task2_answer_en.txt"
normalizer = EnglishTextNormalizer({})

# --- 載入模型和 Tokenizer (調整過的區塊) ---
try:
    print(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, add_prefix_space=True)

    print(f"Loading config from {MODEL_PATH}...")
    # 此 config 物件將包含 num_labels, id2label, label2id, 以及用於 backbone 的 name_or_path
    config = AutoConfig.from_pretrained(MODEL_PATH)

    print(f"Initializing TokenClassificationWithCRF model with config...")
    # 先使用載入的 config 初始化模型實例
    model = TokenClassificationWithCRF(config=config)

    # 載入模型權重
    model_weights_path_pt = os.path.join(MODEL_PATH, "pytorch_model.bin")
    model_weights_path_sf = os.path.join(MODEL_PATH, "model.safetensors")

    if os.path.exists(model_weights_path_pt):
        print(f"Loading state_dict from {model_weights_path_pt}...")
        model.load_state_dict(torch.load(model_weights_path_pt, map_location="cpu"))
    elif os.path.exists(model_weights_path_sf):
        print(f"Loading state_dict from {model_weights_path_sf} (safetensors)...")
        from safetensors.torch import load_file as load_safetensors_file

        model.load_state_dict(load_safetensors_file(model_weights_path_sf))
    else:
        print(f"Warning: No standard 'pytorch_model.bin' or 'model.safetensors' found directly in {MODEL_PATH}.")
        print(
            "Falling back to PreTrainedModel.from_pretrained, which might re-download backbone or require specific file naming.")
        # 作為備案，如果沒有找到明確的權重檔，嘗試使用 from_pretrained 載入
        # 注意：如果 MODEL_PATH 下的檔案命名不符合 from_pretrained 的預期，仍可能失敗
        model = TokenClassificationWithCRF.from_pretrained(MODEL_PATH, config=config)

    model.to(DEVICE)
    model.eval()
    print(f"Model and tokenizer loaded successfully from {MODEL_PATH} on {DEVICE}.")
    print(f"Model backbone is expected to be: {model.config.name_or_path}")  # 來自儲存的 config.json

except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    import traceback

    traceback.print_exc()
    exit()

# 從模型配置中獲取 id2label
id2label = model.config.id2label
if id2label is None:
    print("Error: id2label is None in model config. Cannot proceed.")
    exit()
# 確保 id2label 的鍵是整數 (從 JSON 載入時可能是字串)
try:
    id2label = {int(k): v for k, v in id2label.items()}
except ValueError as e:
    print(f"Error converting id2label keys to int: {e}. Current id2label: {id2label}")
    print("Assuming keys might already be int if direct attribute access, or mixed. Trying to filter.")
    temp_id2label = {}
    valid_conversion = True
    for k, v in id2label.items():
        try:
            temp_id2label[int(k)] = v
        except ValueError:
            if isinstance(k, int):  # Key is already an int
                temp_id2label[k] = v
            else:
                print(f"Cannot convert key {k} to int.")
                valid_conversion = False
                break
    if valid_conversion:
        id2label = temp_id2label
    else:
        print("Failed to normalize id2label keys. Check config format.")
        exit()

label_list = list(id2label.values())

label2id = model.config.label2id
if label2id is None:
    print("Error: label2id is None in model config. Cannot proceed.")
    exit()

print("Available labels (from id2label mapping):", label_list)

# 為 CRF 解碼後的填充找一個合適的標籤 ID
if "O" in label2id:
    PAD_LABEL_ID = label2id["O"]
    print(f"Using 'O' label ID ({PAD_LABEL_ID}) for padding decoded sequences.")
elif 0 in id2label:  # Fallback if 'O' is not in label2id but 0 is a valid label ID
    PAD_LABEL_ID = 0
    print(f"Warning: 'O' label not in label2id. Using label ID 0 ({id2label[0]}) for padding decoded sequences.")
else:
    # If 'O' is not available and 0 is not a valid key, this is problematic.
    # Try to find any valid label ID or default to 0 if no other choice.
    if id2label:  # Check if id2label is not empty
        PAD_LABEL_ID = next(iter(id2label.keys()))  # Get the first available label ID
        print(
            f"Warning: 'O' label not in label2id and 0 is not mapped. Using first available label ID {PAD_LABEL_ID} ('{id2label[PAD_LABEL_ID]}') for padding.")
    else:  # id2label is empty, very problematic
        PAD_LABEL_ID = 0  # Default to 0 if id2label is empty
        print(
            f"CRITICAL Warning: 'O' label not in label2id, 0 is not mapped, and id2label is empty. Defaulting PAD_LABEL_ID to 0. This may lead to errors.")


def predict_ner(text):
    """
    對輸入的文本進行NER預測。
    假設 text 中的詞是以空格分隔的。
    """
    if not text.strip():
        return []

    words = text.split()
    if not words:
        return []

    # 當 is_split_into_words=True 時，tokenizer 會直接將 'words' 視為預先分詞的結果。
    # return_offsets_mapping=False 是為了簡化，因為本範例的後處理不直接使用 offset。
    tokenized_inputs = tokenizer(
        words,
        truncation=True,
        is_split_into_words=True,
        return_tensors="pt",
        padding="max_length",
        max_length=512,  # 確保與訓練時的 max_length 一致或合理
        return_offsets_mapping=False  # 不返回偏移映射，如果需要精確的字元級實體，則需要設為 True
    )

    # 獲取 word_ids，用於將 token 級別的預測對齊回原始單詞
    word_ids_for_batch = tokenized_inputs.word_ids(batch_index=0)
    inputs_on_device = {k: v.to(DEVICE) for k, v in tokenized_inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs_on_device)
        emissions = outputs.logits  # CRF 輸入需要 logits

        attention_mask = inputs_on_device.get("attention_mask")
        if attention_mask is None:
            # 如果沒有 attention_mask，假設所有 token 都有效
            attention_mask = torch.ones_like(emissions[..., 0], dtype=torch.long, device=emissions.device)
        crf_mask = attention_mask.bool()  # CRF 的 mask 必須是布林型態

        # CRF 解碼，返回最佳路徑 (list of tag IDs)
        decoded_predictions_list = model.crf.decode(emissions=emissions, mask=crf_mask)

        # 由於 CRF.decode 返回的是實際解碼的序列長度，可能比 max_length 短，需要填充
        max_len = emissions.size(1)  # 模型輸出 (logits) 的序列長度
        padded_predictions_for_batch_item = []

        if decoded_predictions_list and decoded_predictions_list[0] is not None:
            p_seq = decoded_predictions_list[0]
            padding_needed = max_len - len(p_seq)
            if padding_needed < 0:
                padding_needed = 0
                p_seq = p_seq[:max_len]  # 如果因某種原因過長，則截斷
            padded_predictions_for_batch_item = p_seq + [PAD_LABEL_ID] * padding_needed
        else:
            # 如果 CRF 解碼失敗或返回空，則全部填充
            padded_predictions_for_batch_item = [PAD_LABEL_ID] * max_len
            print("CRF 解碼失敗")
            if not text.isspace() and text:  # 避免為空或只有空白字串的輸入列印警告
                print(f"Warning: CRF decode returned empty or None for text: '{text[:50]}...'")

        predicted_token_ids_for_batch = padded_predictions_for_batch_item

    # 將 token 級別的預測對齊到原始單詞級別
    word_idx_to_label_id = {}
    for token_idx, current_word_id_in_words_list in enumerate(word_ids_for_batch):
        if current_word_id_in_words_list is None:  # 特殊 token (如 [CLS], [SEP]) 沒有對應的 word_id
            continue
        # 對於每個單詞，我們只取它第一個子 token 的預測標籤。
        # 在 B-I-O 標籤方案中，一個單詞對應多個子 token 時，通常只關注第一個子 token 的標籤。
        # 如果第一個子 token 是 B- 或 I-，則整個單詞被視為該實體的一部分。
        if current_word_id_in_words_list not in word_idx_to_label_id:
            if token_idx < len(predicted_token_ids_for_batch):
                predicted_label_id = predicted_token_ids_for_batch[token_idx]
                if predicted_label_id not in id2label:
                    print(
                        f"Warning: Predicted label ID {predicted_label_id} not in id2label map. Defaulting to PAD_LABEL_ID for word ID {current_word_id_in_words_list}.")
                    word_idx_to_label_id[current_word_id_in_words_list] = PAD_LABEL_ID
                else:
                    word_idx_to_label_id[current_word_id_in_words_list] = predicted_label_id
            else:
                # 預測序列比 word_ids 短的情況 (理論上不應發生，因為已經填充了)
                word_idx_to_label_id[current_word_id_in_words_list] = PAD_LABEL_ID

    aligned_results = []
    for i, word_text in enumerate(words):
        if i in word_idx_to_label_id:
            label_id = word_idx_to_label_id[i]
            if label_id not in id2label:  # 額外防護，確保獲取的 ID 存在於映射中
                print(
                    f"Error: Label ID {label_id} for word '{word_text}' not in id2label map during final alignment. Using PAD_LABEL.")
                aligned_results.append((word_text, id2label.get(PAD_LABEL_ID, "O")))  # Fallback to "O" string
            else:
                aligned_results.append((word_text, id2label[label_id]))
        else:
            # 如果某個單詞沒有對應到任何 token_id (例如，tokenization 錯誤導致的空單詞)
            # 或者沒有在 word_idx_to_label_id 中被賦值 (不應該發生)
            aligned_results.append((word_text, id2label.get(PAD_LABEL_ID, "O")))

    return aligned_results


# --- 主執行緒 (main execution block) ---
if __name__ == "__main__":
    try:
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input JSON file not found at {INPUT_JSON_PATH}")
        exit()
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {INPUT_JSON_PATH}")
        exit()

    all_output_lines = []

    for file_name, entry in data.items():
        original_word_timestamps = entry.get("wordTimeStamp")

        if not isinstance(original_word_timestamps, list):
            print(f"Warning: Skipping {file_name} due to missing or invalid wordTimeStamp (not a list or None).")
            continue

        if not original_word_timestamps:
            print(f"Warning: Skipping {file_name} because wordTimeStamp is empty.")
            continue

        filtered_word_timestamps = []
        filtered_words_for_transcription = []

        for ts_entry in original_word_timestamps:
            text_from_ts = ts_entry.get("Text")
            if text_from_ts is None:
                continue
            normalized_text = normalizer(text_from_ts)
            # 只保留經過正規化後非空且非純空白的單詞
            if normalized_text and normalized_text.strip():
                filtered_word_timestamps.append(ts_entry)
                # 使用原始的 text_from_ts 進行拼接，因為 predict_ner 內部會再處理分詞
                filtered_words_for_transcription.append(text_from_ts)

        if not filtered_word_timestamps:  # 或 not filtered_words_for_transcription
            print(
                f"Warning: Skipping {file_name} as all words were filtered out after normalization or resulted in no words for transcription.")
            continue

        word_timestamps = filtered_word_timestamps
        transcription = " ".join(filtered_words_for_transcription)

        if not transcription.strip():  # 如果轉錄結果在 join 後仍為空 (例如所有單詞都是空字串)
            print(f"Warning: Skipping {file_name} as transcription is empty after processing word timestamps.")
            continue

        predicted_word_labels = predict_ner(transcription)

        # 這裡會檢查對齊後的單詞數量與原始過濾後的 word_timestamps 數量是否一致
        # 如果不一致，表示在 tokenization 或預測對齊過程中出現了問題，跳過該檔案
        if len(predicted_word_labels) != len(word_timestamps):
            print(f"CRITICAL Warning: Word count mismatch for {file_name} AFTER filtering and prediction. "
                  f"Predicted labels: {len(predicted_word_labels)} ({[p[0] for p in predicted_word_labels][:10]}...), "
                  f"Filtered Timestamps: {len(word_timestamps)} ({[wts['Text'] for wts in word_timestamps][:10]}...). ")
            print(f"Transcription ({len(transcription.split())} words): '{transcription[:100]}...'")
            print(f"Skipping {file_name} due to length mismatch after prediction and filtering.")
            continue

        output_for_file = []
        i = 0
        n_words = len(predicted_word_labels)

        while i < n_words:
            current_word_text_from_pred = predicted_word_labels[i][0]  # 來自 predict_ner 內部的分詞結果
            current_label = predicted_word_labels[i][1]

            ts_entry = word_timestamps[i]
            text_from_ts = ts_entry["Text"]  # 來自原始 timestamp 資料的單詞
            current_start_time = ts_entry["startTime"]
            current_end_time = ts_entry["endTime"]

            # B-I-O 實體組合邏輯
            if current_label.startswith("B-"):
                entity_type = current_label[2:]
                entity_texts_list = [text_from_ts]  # 使用來自 timestamp 的原始文字
                entity_start_time = current_start_time
                entity_end_time = current_end_time

                j = i + 1
                while j < n_words:
                    next_label = predicted_word_labels[j][1]
                    next_ts_entry = word_timestamps[j]
                    # 檢查下一個單詞是否為當前實體類型的內部 (I-) 標籤
                    if next_label == f"I-{entity_type}":
                        entity_texts_list.append(next_ts_entry["Text"])  # 使用來自 timestamp 的原始文字
                        entity_end_time = next_ts_entry["endTime"]
                        j += 1
                    else:
                        break  # 如果不是 I- 或類型不匹配，則實體結束
                full_entity_text = " ".join(entity_texts_list)
                output_for_file.append(
                    f"{file_name}\t{entity_type}\t{entity_start_time}\t{entity_end_time}\t{full_entity_text}")
                i = j  # 將索引跳到當前實體的結束位置
            elif current_label.startswith("I-"):
                # 如果遇到單獨的 I- 標籤，這通常表示 B- 標籤被遺漏了，
                # 但為避免丟失資料，這裡仍然將其作為一個單獨的實體輸出。
                # 實體類型直接從 I- 標籤中提取。
                entity_type = current_label[2:]
                output_for_file.append(
                    f"{file_name}\t{entity_type}\t{current_start_time}\t{current_end_time}\t{text_from_ts}")  # 使用來自 timestamp 的原始文字
                i += 1
            else:  # current_label is "O" (或任何非 B-I- 的標籤)
                i += 1  # 繼續處理下一個單詞
        all_output_lines.extend(output_for_file)

    try:
        os.makedirs(os.path.dirname(OUTPUT_TXT_PATH), exist_ok=True)
        with open(OUTPUT_TXT_PATH, 'w', encoding='utf-8') as f_out:
            for line in all_output_lines:
                f_out.write(line + "\n")
        print(f"Successfully wrote predictions to {OUTPUT_TXT_PATH}")
    except IOError:
        print(f"Error: Could not write to output file {OUTPUT_TXT_PATH}")

    print("Prediction process complete.")