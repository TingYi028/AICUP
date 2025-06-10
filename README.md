# ASR 指南

## 環境說明

本專案需分別建立兩個 Python 環境，因 Nemo 與 Whisper 相關套件互斥，請分開安裝：

### 1. Nemo 環境
安裝指令：
```bash
pip install -U nemo_toolkit["asr"]
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install seqeval datasets pytorch-crf
```

### 2. Whisper 環境
安裝指令：
```bash
pip install -r requirements_Whisper.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> ⚠️ **貼心提醒：** 除非執行 Whisper 相關功能，否則所有操作皆於 Nemo 環境進行，僅在需要使用 Whisper 時才需切換至 Whisper 環境。

## 資料前處理

### 1. 去除背景音樂

使用 Ultimate Vocal Remover 軟體去除音訊中的背景音樂。

#### 主要設定
- **處理方法 (CHOOSE PROCESS METHOD)**: MDX-Net
- **分段大小 (SEGMENT SIZE)**: 256
- **重疊 (OVERLAP)**: Default
- **輸出格式 (Output Format)**: WAV

#### 模型與選項
- **選擇 MDX-NET 模型 (CHOOSE MDX-NET MODEL)**: UVR-MDX-NET Inst HQ 3

#### 勾選項目
- ✅ GPU Conversion (GPU 轉換)
- ⬜ Instrumental Only (僅伴奏) - 未勾選
- ✅ Vocals Only (僅人聲)
- ⬜ Sample Mode (30s) (樣本模式 (30秒)) - 未勾選

### 重新命名檔案
將軟體產生的檔案名稱從 `{id}_{origName}_(Vocals)` 轉換回 `{origName}`

### 2. 分離中文與英文音檔

使用語言檢測模型將音檔分離成中文和英文兩個部分。

#### 修改 lang_detect.py 變數
1. **AUDIO_DIR**: 去除背景音樂的音訊資料夾路徑
2. **CHINESE_DIR**: 中文音檔輸出路徑
3. **ENGLISH_DIR**: 英文音檔輸出路徑

#### 執行指令
```bash
cd AICUP/ASR
python lang_detect.py
```

### 3. 轉換音訊格式 & 產生 Manifest 檔案 （只有英文音檔，中文不做訓練）

將音訊轉換為 16kHz 單聲道格式並產生訓練所需的 manifest 檔案。

#### 設定 createManifest.py 變數
1. **AUDIO_DIR**: 去除背景音樂的音訊資料夾路徑
2. **TRANSCRIPT_ORIGINAL**: task1_answer.txt 路徑
3. **MANIFEST_OUTPUT**: 輸出 MANIFEST 路徑
4. **CONVERTED_AUDIO_DIR**: 輸出 16k 音訊路徑

#### 執行指令
```bash
python AICUP/datasets/createManifest.py
```

## 英文 ASR 訓練 (我們沒有訓練中文的ASR)

### 1. 設定訓練配置
修改 `AICUP/ASR/train_config/speech_to_text_finetune.yaml` 檔案中的 `train_ds` 和 `validation_ds` 的 `manifest_filepath`
### 2. 設定訓練配置
下載模型 https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/blob/main/parakeet-tdt-0.6b-v2.nemo
放置在 AICUP/ASR 底下
### 3. 執行訓練
```bash
cd AICUP/ASR
python speech_to_text_finetune.py
```

## 英文 ASR 預測

### 1. 設定預測參數
在執行預測前，需要設定以下變數：

```python
parser.add_argument('--audio_dir', default="../datasets/Test/audio_16k", type=str, help='音頻文件目錄路徑')
parser.add_argument('--output', type=str, help='詳細輸出文件路徑')
parser.add_argument('--labels', default="../datasets/val/task1_answer.txt", type=str, help='標籤文件路徑（用於計算MER）若沒有可以不填')
parser.add_argument('--output_label', type=str, default='Test_result/task1_answer_noFinetuned.txt', help='輸出標籤文件路徑（預設為output_label.txt）')
parser.add_argument('--json_output', default="Test_result/task1_answer_noFinetuned_timeStamp.json", type=str, help='JSON時間戳輸出文件路徑')
parser.add_argument('--model', type=str, default=r'C:\Users\C110151154\PycharmProjects\NeMo\AICUP\ASR\fintuned_model\Speech_To_Text_Finetuning_train80_val20.nemo', help='模型名稱，若要使用不微調版本使用 AICUP/ASR/parakeet-tdt-0.6b-v2.nemo')
```

### 2. 執行預測
```bash
python AICUP/ASR/parakeet_asr_predictor.py
```

## 中文 ASR 預測

cd AICUP/ChineseASR
### 1. 設定 predict.py 參數
在執行預測前，需要設定以下變數：

```python
VAL_AUDIO_DATA_DIR ="要預測的音檔資料夾"
VAL_LABEL_FILE_PATH="要預測的音檔Label"
OUTPUT_FILE_PATH="輸出task1_answer.txt路徑"
OUTPUT_JSON_FILE_PATH="輸出timestamp的路徑"
```

### 2. 執行預測
```bash
cd AICUP/ChineseASR
python predict.py

```

## 整合英文與中文輸出結果

