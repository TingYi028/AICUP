import json
import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F, CrossEntropyLoss

from datasets import load_dataset, Features, Value, Sequence
from transformers import (
    AutoConfig,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    # 直接使用 Hugging Face 的標準模型
    AutoModelForTokenClassification, PreTrainedModel, AutoModel
)
from transformers.modeling_outputs import TokenClassifierOutput

# 加入seqeval用於NER評估
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 1. 全局變數，用於 compute_metrics
_ENTITY_TYPES_FOR_METRICS = []
_ID2LABEL_FOR_METRICS = {}

# 2. 自訂的 Token Classification 模型
class TokenClassification(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # 使用 AutoModel 載入預訓練模型
        self.backbone = AutoModel.from_pretrained(
            config.name_or_path,
            config=config,
            trust_remote_code=True,
        )
        self.backbone.gradient_checkpointing_enable()

        # 分類層設定
        classifier_dropout = getattr(config, 'classifier_dropout', None)
        if classifier_dropout is None:
            classifier_dropout = getattr(config, 'hidden_dropout_prob', 0.1)

        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fct = CrossEntropyLoss()

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

        # 通過主幹網路獲取特徵
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
            **kwargs
        )

        sequence_output = outputs.last_hidden_state  # last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # 發射分數

        loss = None
        if labels is not None:
            # 初始化標準的交叉熵損失函數
            # ignore_index=-100 會自動忽略那些不需要計算損失的 token (例如 padding 或 [CLS], [SEP] 的子詞)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# 3. 使用模型內部 CRF 損失的 Trainer
class SimpleCRFTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def filter_out_entity_types(labels, predictions, entity_types_to_remove):
    """
    過濾掉指定的實體類型
    """
    filtered_labels = []
    filtered_predictions = []

    for label_seq, pred_seq in zip(labels, predictions):
        filtered_label_seq = []
        filtered_pred_seq = []

        for label, pred in zip(label_seq, pred_seq):
            # 檢查標籤是否屬於要過濾掉的實體類型
            if label in entity_types_to_remove:
                filtered_label_seq.append('O')
            else:
                filtered_label_seq.append(label)
            if pred in entity_types_to_remove:
                filtered_pred_seq.append('O')
            else:
                filtered_pred_seq.append(pred)

        filtered_labels.append(filtered_label_seq)
        filtered_predictions.append(filtered_pred_seq)

    return filtered_labels, filtered_predictions


# 2. 評估函數 (原 compute_metrics_for_crf，邏輯不變，適用於argmax)
def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # 將 predictions 進行 argmax (如果模型輸出 logits)
    # Trainer 預設會做 argmax，所以這裡的 predictions 已經是預測的 index
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [_ID2LABEL_FOR_METRICS.get(p_idx, 'O') for p_idx, l_idx in zip(p_row, l_row) if l_idx != -100]
        for p_row, l_row in zip(predictions, labels)
    ]
    true_labels = [
        [_ID2LABEL_FOR_METRICS.get(l_idx, 'O') for l_idx in l_row if l_idx != -100]
        for l_row in labels
    ]

    precision = precision_score(true_labels, true_predictions, zero_division=0)
    recall = recall_score(true_labels, true_predictions, zero_division=0)
    f1 = f1_score(true_labels, true_predictions, zero_division=0, average='macro')
    accuracy = accuracy_score(true_labels, true_predictions)

    report = classification_report(true_labels, true_predictions, output_dict=True, zero_division=0)

    print("考慮所有類別: ")
    print(classification_report(true_labels, true_predictions, zero_division=0))

    results = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }

    # 過濾掉指定的實體類型
    filtered_labels, filtered_predictions = filter_out_entity_types(
        true_labels, true_predictions, ["B-COUNTY", "I-COUNTY", "B-PROFESSION", "I-PROFESSION",
                                        "B-ORGANIZATION", "I-ORGANIZATION", "B-DISTRICT", "I-DISTRICT",
                                        "B-COUNTRY", "I-COUNTRY", "B-LOCATION-OTHER", "I-LOCATION-OTHER", ]
    )
    print("不考慮極少數類別: ")
    print(classification_report(filtered_labels, filtered_predictions, zero_division=0))

    # 計算各實體類型的指標
    for entity_type in _ENTITY_TYPES_FOR_METRICS:
        # report中的key是實體類型名稱，而不是B-/I-前綴
        if entity_type in report:
            results[f"{entity_type}_precision"] = report[entity_type]["precision"]
            results[f"{entity_type}_recall"] = report[entity_type]["recall"]
            results[f"{entity_type}_f1-score"] = report[entity_type]["f1-score"]
            results[f"{entity_type}_support"] = report[entity_type]["support"]

    return results


# 3. Main 函數
def main():
    # --- 配置 ---
    base_model = "Qwen/Qwen3-Embedding-0.6B"
    train_file_path = r"./datasets/train_bio.json"
    eval_file_path = r"./datasets/val_bio.json"
    output_dir = "./results_ner_Qwen/Qwen3-Embedding-0.6B"  # 移除-crf後綴

    # --- 定義自訂標籤 ---
    entity_types = [
        "PATIENT", "DOCTOR", "USERNAME", "FAMILYNAME", "PERSONALNAME",
        "PROFESSION", "ROOM", "DEPARTMENT", "HOSPITAL", "ORGANIZATION",
        "STREET", "CITY", "DISTRICT", "COUNTY", "STATE", "COUNTRY", "ZIP",
        "LOCATION-OTHER", "AGE", "DATE", "TIME", "DURATION", "SET",
        "PHONE", "FAX", "EMAIL", "URL", "IPADDRESS",
        "SOCIAL_SECURITY_NUMBER", "MEDICAL_RECORD_NUMBER", "HEALTH_PLAN_NUMBER",
        "ACCOUNT_NUMBER", "LICENSE_NUMBER", "VEHICLE_ID", "DEVICE_ID",
        "BIOMETRIC_ID", "ID_NUMBER"
    ]

    entity_types = [label.strip() for label in entity_types]
    label_list = ["O"]
    for entity in entity_types:
        label_list.append(f"B-{entity}")
        label_list.append(f"I-{entity}")

    print("使用自訂標籤列表:", len(label_list), "個標籤")
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    # 設定全局變數以供 compute_metrics 使用
    global _ENTITY_TYPES_FOR_METRICS, _ID2LABEL_FOR_METRICS
    _ENTITY_TYPES_FOR_METRICS = entity_types
    _ID2LABEL_FOR_METRICS = id2label

    # --- 定義 Dataset 特徵 ---
    features = Features({
        'id': Value('int64'),
        'tokens': Sequence(Value('string')),
        'labels': Sequence(Value('string'))
    })

    # --- 載入資料集 ---
    try:
        raw_datasets = load_dataset(
            'json',
            data_files={'train': train_file_path, 'validation': eval_file_path},
            features=features
        )
        print(f"成功載入資料集: 訓練集 {len(raw_datasets['train'])} 筆, 驗證集 {len(raw_datasets['validation'])} 筆")
    except Exception as e:
        print(f"載入資料集時發生錯誤: {e}")
        print("嘗試不使用明確特徵載入資料集...")
        raw_datasets = load_dataset(
            'json',
            data_files={'train': train_file_path, 'validation': eval_file_path}
        )

    # --- 初始化 Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side='left')
    eod_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    # --- Tokenize 和對齊標籤的函數 ---
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding="max_length",
            max_length=200
        )

        # 這個操作可能特定於Qwen模型，予以保留
        for seq, att in zip(tokenized_inputs["input_ids"], tokenized_inputs["attention_mask"]):
            seq.append(eod_id)
            att.append(1)

        labels = []
        for i, label_sequence in enumerate(examples["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:
                if word_idx is None:  # 特殊標記
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # 新詞的第一個標記
                    original_label = label_sequence[word_idx]
                    label_ids.append(label2id.get(original_label, label2id['O']))
                else:  # 同一詞的後續標記
                    original_label = label_sequence[word_idx]
                    if original_label.startswith("B-"):
                        original_label = "I-" + original_label[2:]
                    label_ids.append(label2id.get(original_label, label2id['O']))
                previous_word_idx = word_idx

            label_ids.append(-100)
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_datasets = raw_datasets.map(tokenize_and_align_labels, batched=True)

    # --- Data Collator ---
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # --- 初始化標準模型 ---
    print(f"初始化標準模型: AutoModelForTokenClassification from {base_model}")
    config = AutoConfig.from_pretrained(
        base_model,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        trust_remote_code=True,
    )
    config.name_or_path = base_model
    model = TokenClassification(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"使用裝置: {device}")

    # --- 訓練參數 ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=15,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        lr_scheduler_type="cosine",
        eval_on_start=True,
        dataloader_pin_memory=True,
        bf16=True
    )

    # --- Trainer 初始化 (使用標準 Trainer) ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,  # tokenizer 參數建議傳入
        data_collator=data_collator,
        compute_metrics=compute_metrics  # 使用更名後的評估函數
    )

    # --- 開始訓練 ---
    print("開始訓練...")
    trainer.train()

    # --- 評估模型 ---
    print("評估模型...")
    eval_results = trainer.evaluate()
    print(f"評估結果: {eval_results}")

    # --- 儲存模型和 Tokenizer ---
    print(f"儲存模型至 {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("訓練完成。")


# 4. 程式進入點
if __name__ == "__main__":
    main()