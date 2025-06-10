# ASR & NER 開發指南

## 📖 概述

本專案包含自動語音識別 (ASR) 和命名實體識別 (NER) 兩個主要功能模組。

---

## 🔧 環境設定

本專案需分別建立兩個 Python 環境，因 Nemo 與 Whisper 相關套件互斥，請分開安裝：

### 環境 1: Nemo 環境
```bash
pip install -U nemo_toolkit["asr"]
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install seqeval datasets pytorch-crf
```

### 環境 2: Whisper 環境
```bash
pip install -r requirements_Whisper.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> ⚠️ **重要提醒**  
> 除非執行 Whisper 相關功能，否則所有操作皆於 Nemo 環境進行，僅在需要使用 Whisper 時才需切換至 Whisper 環境。

---

## 🎵 ASR 自動語音識別

### 步驟 1: 資料前處理

#### 1.1 去除背景音樂

使用 **Ultimate Vocal Remover** 軟體去除音訊中的背景音樂。

**主要設定：**
- **處理方法**: MDX-Net
- **分段大小**: 256
- **重疊**: Default
- **輸出格式**: WAV

**模型選擇：**
- **MDX-NET 模型**: UVR-MDX-NET Inst HQ 3

**選項設定：**
- ✅ GPU Conversion (GPU 轉換)
- ⬜ Instrumental Only (僅伴奏)
- ✅ Vocals Only (僅人聲)
- ⬜ Sample Mode (30s) (樣本模式)

**重新命名檔案：**  
將軟體產生的檔案名稱從 `{id}_{origName}_(Vocals)` 轉換回 `{origName}`

#### 1.2 分離中文與英文音檔

使用語言檢測模型將音檔分離成中文和英文兩個部分。

**修改 `lang_detect.py` 變數：**
1. `AUDIO_DIR`: 去除背景音樂的音訊資料夾路徑
2. `CHINESE_DIR`: 中文音檔輸出路徑
3. `ENGLISH_DIR`: 英文音檔輸出路徑

**執行指令：**
```bash
cd AICUP/ASR
python lang_detect.py
```

#### 1.3 轉換音訊格式 & 產生 Manifest 檔案

> **注意：** 僅處理英文音檔，中文不做訓練

將音訊轉換為 16kHz 單聲道格式並產生訓練所需的 manifest 檔案。

**設定 `createManifest.py` 變數：**
1. `AUDIO_DIR`: 去除背景音樂的音訊資料夾路徑
2. `TRANSCRIPT_ORIGINAL`: task1_answer.txt 路徑
3. `MANIFEST_OUTPUT`: 輸出 MANIFEST 路徑
4. `CONVERTED_AUDIO_DIR`: 輸出 16k 音訊路徑

**執行指令：**
```bash
cd AICUP/datasets
python createManifest.py
```

### 步驟 2: 英文 ASR 訓練

#### 2.1 設定訓練配置
修改 `AICUP/ASR/train_config/speech_to_text_finetune.yaml` 檔案中的 `train_ds` 和 `validation_ds` 的 `manifest_filepath`

#### 2.2 下載預訓練模型
1. 下載模型：[parakeet-tdt-0.6b-v2.nemo](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/blob/main/parakeet-tdt-0.6b-v2.nemo)
2. 放置在 `AICUP/ASR` 目錄下

#### 2.3 執行訓練
```bash
cd AICUP/ASR
python speech_to_text_finetune.py
```

### 步驟 3: 英文 ASR 預測

#### 3.1 設定預測參數
```python
parser.add_argument('--audio_dir', default="../datasets/Test/audio_16k", type=str, help='音頻文件目錄路徑')
parser.add_argument('--output', type=str, help='詳細輸出文件路徑')
parser.add_argument('--labels', default="../datasets/val/task1_answer.txt", type=str, help='標籤文件路徑（用於計算MER）若沒有可以不填')
parser.add_argument('--output_label', type=str, default='Test_result/task1_answer_noFinetuned.txt', help='輸出標籤文件路徑（預設為output_label.txt）')
parser.add_argument('--json_output', default="Test_result/task1_answer_noFinetuned_timeStamp.json", type=str, help='JSON時間戳輸出文件路徑')
parser.add_argument('--model', type=str, default=r'C:\Users\C110151154\PycharmProjects\NeMo\AICUP\ASR\fintuned_model\Speech_To_Text_Finetuning_train80_val20.nemo', help='模型名稱，若要使用不微調版本使用 AICUP/ASR/parakeet-tdt-0.6b-v2.nemo')
```

#### 3.2 執行預測
```bash
cd AICUP/ASR
python parakeet_asr_predictor.py
```

### 步驟 4: 中文 ASR 預測

#### 4.1 設定 `predict.py` 參數
```python
VAL_AUDIO_DATA_DIR = "要預測的音檔資料夾"
VAL_LABEL_FILE_PATH = "要預測的音檔Label"
OUTPUT_FILE_PATH = "輸出task1_answer.txt路徑"
OUTPUT_JSON_FILE_PATH = "輸出timestamp的路徑"
```

#### 4.2 執行預測
```bash
cd AICUP/ChineseASR
python predict.py
```

### 步驟 5: 整合英文與中文輸出結果

---

## 🏷️ NER 命名實體識別

### 英文 NER 訓練

#### 1. 創建 BIO.json

進入 NER 資料集目錄：
```bash
cd AICUP/NER/Task2/datasets
```

修改 `createBIO.py`，設定以下參數：
```python
bio_json = convert_to_bio_json(
    r'C:\Users\C110151154\PycharmProjects\NeMo\AICUP\datasets\val\task1_answer.txt',
    r'C:\Users\C110151154\PycharmProjects\NeMo\AICUP\datasets\val\task2_answer.txt',
    './val_bio_raw.json'
)
training_data = convert_to_training_format(
    bio_json, 
    './val_bio.json'
)
```

**參數說明：**
- **第一個參數**: task1 的 label 檔案路徑
- **第二個參數**: task2 的 label 檔案路徑  
- **第三個參數**: raw bio 的輸出路徑
- **第四個參數**: 用來訓練的 bio 輸出路徑

#### 2. 修改訓練設定

修改 `train.py` 中的以下參數：
- `train_file_path`: 前面轉換出的 bio 檔案路徑
- `eval_file_path`: 前面轉換出的 bio 檔案路徑
- `output_dir`: 模型權重輸出資料夾路徑

#### 3. 執行訓練
```bash
python train.py
```

### 英文 NER 預測

#### 1. 設定預測參數
修改 `predict.py` 中的以下參數：
- `MODEL_PATH`: CHECKPOINT 的資料夾路徑
- `INPUT_JSON_PATH`: ASR 輸出的帶有 `_timestamp` 後綴的 .json 檔案路徑
- `OUTPUT_TXT_PATH`: 輸出檔案路徑

#### 2. 執行預測
執行 NER 預測腳本

### 中文 NER 訓練

#### 1. 創建 BIO.json

進入中文 NER 資料集目錄：
```bash
cd AICUP/NER/Chinese/datasets
```

修改 `createBIO.py`，設定參數：
```python
bio_json = convert_to_bio_json(
    r'C:\Users\C110151154\PycharmProjects\NeMo\AICUP\datasets\val\task1_answer.txt',
    r'C:\Users\C110151154\PycharmProjects\NeMo\AICUP\datasets\val\task2_answer.txt',
    './val_bio_raw.json'
)
training_data = convert_to_training_format(
    bio_json, 
    './val_bio.json'
)
```

**參數說明：**
- **第一個參數**: task1 的 label 檔案路徑
- **第二個參數**: task2 的 label 檔案路徑  
- **第三個參數**: raw bio 的輸出路徑
- **第四個參數**: 用來訓練的 bio 輸出路徑

#### 2. 修改訓練設定

修改 `train.py` 中的以下參數：
- `train_file_path`: 前面轉換出的 bio 檔案路徑
- `eval_file_path`: 前面轉換出的 bio 檔案路徑
- `output_dir`: 模型權重輸出資料夾路徑

#### 3. 執行訓練
```bash
python train.py
```

### 整合中文與英文 NER 結果

---

## 📝 注意事項

1. **環境管理**: 請確實分離 Nemo 和 Whisper 環境
2. **路徑設定**: 所有檔案路徑請根據實際環境調整
3. **模型下載**: 確保預訓練模型已正確下載並放置在指定位置
4. **資料格式**: 注意音訊檔案格式和採樣率要求
