import json
import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from datasets import load_dataset, Features, Value, Sequence
from transformers import (
    AutoConfig,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    PreTrainedModel,
    AutoModel
)
from transformers.modeling_outputs import TokenClassifierOutput
from torchcrf import CRF

# 加入seqeval用於NER評估
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 1. FocalLoss 類別
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=-100)
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * CE_loss

        # 只對非忽略的標籤計算損失
        active_loss = targets != -100
        F_loss = F_loss[active_loss]

        if F_loss.numel() == 0:  # 如果所有標籤都被忽略
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


# 2. 自訂的 Token Classification 模型，整合 CRF 層
class TokenClassificationWithCRF(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # 使用 AutoModel 載入預訓練模型
        self.backbone = AutoModel.from_pretrained(
            config.name_or_path,
            config=config,
            trust_remote_code=True,
        )

        # 分類層設定
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
            # CRF 期望的 attention_mask 是布林類型
            crf_mask = attention_mask.bool() if attention_mask is not None else torch.ones_like(
                labels, dtype=torch.bool, device=labels.device
            )

            # 處理 -100 標籤（忽略的標籤）
            active_labels_for_crf = labels.clone()
            active_labels_for_crf[labels == -100] = 0  # 替換為有效標籤索引

            # 計算 CRF 負對數似然損失
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


# 3. 使用模型內部 CRF 損失的 Trainer
class SimpleCRFTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


# 4. 覆寫 prediction_step 以進行 CRF 解碼的 Trainer
class TrainerWithCRFDecoding(SimpleCRFTrainer):
    def create_optimizer(self):
        """
        創建自定義optimizer，為CRF層設定更高的學習率
        """
        if self.optimizer is None:
            base_lr = self.args.learning_rate  # 基礎學習率 2e-5
            crf_lr = base_lr * 100  # CRF層學習率增加100倍，即 2e-2

            # 分離CRF參數和其他參數
            crf_params = []
            other_params = []

            for name, param in self.model.named_parameters():
                if 'crf' in name:
                    crf_params.append(param)
                    print(f"CRF參數: {name} - 學習率: {crf_lr}")
                else:
                    other_params.append(param)

            # 創建參數組
            param_groups = [
                {'params': other_params, 'lr': base_lr, 'weight_decay': self.args.weight_decay},
                {'params': crf_params, 'lr': crf_lr, 'weight_decay': self.args.weight_decay}
            ]

            print(f"基礎學習率: {base_lr}")
            print(f"CRF層學習率: {crf_lr}")
            print(f"CRF參數數量: {len(crf_params)}")
            print(f"其他參數數量: {len(other_params)}")

            # 使用AdamW optimizer
            self.optimizer = torch.optim.AdamW(param_groups)

        return self.optimizer

    def prediction_step(
            self,
            model: nn.Module,
            inputs: dict,
            prediction_loss_only: bool,
            ignore_keys: list = None,
    ):
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)

        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            if has_labels:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
                emissions = outputs.logits
            else:
                loss = None
                outputs = model(**inputs, return_dict=True)
                emissions = outputs.logits

        if prediction_loss_only:
            return (loss, None, None)

        # CRF 解碼
        if hasattr(model, 'crf') and callable(getattr(model.crf, 'decode')):
            attention_mask = inputs.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(emissions[..., 0], dtype=torch.long, device=emissions.device)

            crf_mask = attention_mask.bool().to(emissions.device)
            decoded_predictions_list = model.crf.decode(emissions=emissions, mask=crf_mask)

            # 填充預測結果
            max_len = emissions.size(1)
            pad_token_label_id = -100
            padded_predictions = [
                p_seq + [pad_token_label_id] * (max_len - len(p_seq))
                for p_seq in decoded_predictions_list
            ]
            predictions_tensor = torch.tensor(padded_predictions, dtype=torch.long, device=emissions.device)
        else:
            print("Warning: Model does not have a CRF decode method. Falling back to argmax.")
            predictions_tensor = torch.argmax(emissions, dim=-1)

        if has_labels:
            labels_tensor = inputs.get("labels")
            if labels_tensor is not None:
                labels_tensor = labels_tensor.detach()
            return (loss, predictions_tensor, labels_tensor)
        else:
            return (loss, predictions_tensor, None)


# 5. 全局變數，用於 compute_metrics
_ENTITY_TYPES_FOR_METRICS = []
_ID2LABEL_FOR_METRICS = {}


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


# 6. CRF 解碼後的評估函數
def compute_metrics_for_crf(eval_pred):
    predictions, labels = eval_pred

    true_predictions = [
        [_ID2LABEL_FOR_METRICS[p_idx.item()] for p_idx, l_idx in zip(p_row, l_row) if l_idx.item() != -100]
        for p_row, l_row in zip(predictions, labels)
    ]
    true_labels = [
        [_ID2LABEL_FOR_METRICS[l_idx.item()] for l_idx in l_row if l_idx.item() != -100]
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
                                        "B-COUNTRY", "I-COUNTRY", "B-LOCATION-OTHER", "I-LOCATION-OTHER",]
    )
    print("不考慮極少數類別: ")
    print(classification_report(filtered_labels, filtered_predictions, zero_division=0))
    # 計算各實體類型的指標
    for entity_type in _ENTITY_TYPES_FOR_METRICS:
        if entity_type in report:
            results[f"{entity_type}_precision"] = report[entity_type]["precision"]
            results[f"{entity_type}_recall"] = report[entity_type]["recall"]
            results[f"{entity_type}_f1-score"] = report[entity_type]["f1-score"]
            results[f"{entity_type}_support"] = report[entity_type]["support"]

    return results



# 8. Main 函數
def main():
    # --- 配置 ---
    base_model = "microsoft/deberta-v3-large" # deberta-v3-large 的基礎模型名稱
    # model_checkpoint = r"C:\Users\C110151154\PycharmProjects\NeMo\AICUP\NER\ontonotes5\results_ner_ontonotes5_crf\deberta-v3-large\checkpoint-1876\model.safetensors"
    # model_checkpoint = r"C:\Users\C110151154\PycharmProjects\NeMo\AICUP\NER\pii-masking-400k\results_ner_pii_crf\deberta-v3-large\checkpoint-4272\model.safetensors"

    train_file_path = r"./datasets/train_bio.json"
    eval_file_path = r"./datasets/val_bio.json"
    output_dir = "./results_ner_microsoft/deberta-v3-large-crf-test"
    # print("start ontonotes5")
    # print("start pii")

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
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # --- Tokenize 和對齊標籤的函數 ---
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding="max_length",
            max_length=200
        )

        labels = []
        for i, label_sequence in enumerate(examples["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None or word_idx == 151643 or word_idx == 151645:  # 特殊標記
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

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_datasets = raw_datasets.map(tokenize_and_align_labels, batched=True)

    # --- Data Collator ---
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # --- 初始化自訂模型 ---
    print(f"初始化自訂模型: TokenClassificationWithCRF from {base_model}")
    config = AutoConfig.from_pretrained(
        base_model,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        trust_remote_code=True,
    )
    # 設定模型路徑以供 TokenClassificationWithCRF 使用
    config.name_or_path = base_model

    model = TokenClassificationWithCRF(config)

    # # 載入您自訂的 safetensors 權重
    # try:
    #     # 判斷是 safetensors 還是 pytorch_model.bin
    #     if model_checkpoint.endswith(".safetensors"):
    #         from safetensors.torch import load_file
    #         state_dict = load_file(model_checkpoint)
    #         print(f"成功載入 safetensors 權重: {model_checkpoint}")
    #     elif model_checkpoint.endswith(".bin"):
    #         state_dict = torch.load(model_checkpoint, map_location="cpu")
    #         print(f"成功載入 PyTorch bin 權重: {model_checkpoint}")
    #     else:
    #         raise ValueError("不支持的權重檔案格式，請提供 .safetensors 或 .bin 檔案。")
    #     print(state_dict)
    #     print(model)
    #     # 移除 state_dict 中可能存在的 "model." 前綴
    #     new_state_dict = {}
    #     for k, v in state_dict.items():
    #         # 檢查鍵是否以 'model.' 開頭，如果是則移除前綴
    #         if k.startswith("model."):
    #             k = k[len("model."):]
    #
    #         # 略過 CRF 相關的鍵
    #         if k.startswith("crf."):
    #             print(f"跳過 CRF 權重鍵: {k}")
    #             continue
    #         # 略過 Classifier 相關的鍵（因為 size mismatch）
    #         if k.startswith("classifier."):
    #             print(f"跳過 Classifier 權重鍵 (避免 size mismatch): {k}")
    #             continue
    #
    #         new_state_dict[k] = v
    #
    #     # 載入模型狀態字典
    #     model.load_state_dict(new_state_dict, strict=False )
    #     print("自訂 safetensors 權重已成功載入到模型中。")
    # except Exception as e:
    #     print(f"載入自訂 safetensors 權重時發生錯誤: {e}")
    #     print("將繼續使用預訓練模型權重。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"使用裝置: {device}")

    # --- 訓練參數 ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=64,
        num_train_epochs=8,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        lr_scheduler_type="cosine",
        # warmup_ratio=0.1,
        eval_on_start=True,
        dataloader_pin_memory=True,
        bf16=True,
        report_to="none"
    )

    # --- Trainer 初始化 ---
    trainer = TrainerWithCRFDecoding(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_for_crf
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


# 10. 程式進入點
if __name__ == "__main__":
    main()

