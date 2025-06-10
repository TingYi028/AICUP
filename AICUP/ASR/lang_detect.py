import os
import shutil
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from tqdm import tqdm

import librosa

# 參數設定
MODEL_NAME = "openai/whisper-large-v3"
AUDIO_DIR = r"C:\Users\C110151154\PycharmProjects\NeMo\AICUP\datasets\HybridTest\audio_16k"
CHINESE_DIR = r"C:\Users\C110151154\PycharmProjects\NeMo\AICUP\ASR\Split_wav\zh"
ENGLISH_DIR = r"C:\Users\C110151154\PycharmProjects\NeMo\AICUP\ASR\Split_wav\en"
AUDIO_EXT = ".wav"

os.makedirs(CHINESE_DIR, exist_ok=True)
os.makedirs(ENGLISH_DIR, exist_ok=True)

# 設定運算裝置與型態
device = "cuda:0"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# 載入模型與處理器
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_NAME, torch_dtype=torch_dtype
)
model.to(device)
processor = AutoProcessor.from_pretrained(MODEL_NAME)

# 初始化 pipeline
asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    chunk_length_s=30,
)

# 判斷中文字比例的函數
def chinese_ratio(text):
    chinese_count = sum('\u4e00' <= ch <= '\u9fff' for ch in text)
    total_count = len(text)
    if total_count == 0:
        return 0
    return chinese_count / total_count

# 處理每個音檔
for fname in tqdm(os.listdir(AUDIO_DIR)):
    if not fname.lower().endswith(AUDIO_EXT):
        continue
    wav_path = os.path.join(AUDIO_DIR, fname)
    try:
        # 只偵測語言，不需完整轉錄
        # 先取前10秒片段以加速語言偵測
        transcribe_10 = asr_pipe(wav_path, generate_kwargs={"task": "transcribe", "return_timestamps": False, "max_new_tokens": 10})['text']
        ratio = chinese_ratio(transcribe_10)
        # 分類存放
        if ratio > 0.5:
            print(f"{fname}: zh")
            shutil.copy(wav_path, os.path.join(CHINESE_DIR, fname))
        else:
            print(f"{fname}: en")
            shutil.copy(wav_path, os.path.join(ENGLISH_DIR, fname))
    except Exception as e:
        print(f"處理 {fname} 時發生錯誤: {e}")

print("分割完成！")
