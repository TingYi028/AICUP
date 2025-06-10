import os
import json
import soundfile as sf
import subprocess
import shutil
from pathlib import Path
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

# 初始化文本正規化器
normalizer = EnglishTextNormalizer({})

def convert_audio_to_16k_mono(input_file, output_file):
    """
    使用FFmpeg將音訊檔案轉換為16kHz單聲道WAV格式
    """
    try:
        # 轉換為絕對路徑
        input_file_abs = os.path.abspath(input_file)
        output_file_abs = os.path.abspath(output_file)

        # 確保輸出目錄存在
        os.makedirs(os.path.dirname(output_file_abs), exist_ok=True)

        # FFmpeg命令：轉換為16kHz單聲道WAV
        command = [
            'ffmpeg', '-y',  # -y 表示覆蓋輸出檔案
            '-i', input_file_abs,
            '-acodec', 'pcm_s16le',  # 16位元PCM編碼
            '-ac', '1',  # 單聲道
            '-ar', '16000',  # 16kHz採樣率
            output_file_abs
        ]

        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode == 0:
            return True
        else:
            print(f"FFmpeg錯誤: {result.stderr}")
            return False

    except Exception as e:
        print(f"轉換音訊檔案時發生錯誤: {e}")
        return False


def batch_convert_audio(input_dir, output_dir, supported_formats=None):
    """
    批次轉換資料夾中的音訊檔案為16kHz單聲道WAV格式
    """
    if supported_formats is None:
        supported_formats = ['.mp3', '.m4a', '.aac', '.flac', '.wav', '.ogg', '.wma']

    converted_files = {}
    failed_files = []

    # 轉換為絕對路徑
    input_dir_abs = os.path.abspath(input_dir)
    output_dir_abs = os.path.abspath(output_dir)

    print(f"開始批次轉換音訊檔案從 '{input_dir_abs}' 到 '{output_dir_abs}'...")

    if not os.path.exists(input_dir_abs):
        print(f"錯誤：輸入目錄 '{input_dir_abs}' 不存在。")
        return converted_files, failed_files

    # 檢查FFmpeg是否可用
    if not shutil.which('ffmpeg'):
        print("錯誤：找不到FFmpeg。請確保已安裝FFmpeg並添加到系統PATH中。")
        return converted_files, failed_files

    # 遍歷輸入目錄中的所有檔案
    for filename in os.listdir(input_dir_abs):
        file_path = os.path.join(input_dir_abs, filename)

        # 跳過目錄
        if os.path.isdir(file_path):
            continue

        # 檢查檔案副檔名
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in supported_formats:
            continue

        # 產生輸出檔案路徑（絕對路徑）
        filename_stem = os.path.splitext(filename)[0]
        output_filename = f"{filename_stem}.wav"
        output_path = os.path.join(output_dir_abs, output_filename)

        print(f"轉換: {filename} -> {output_filename}")

        # 執行轉換
        if convert_audio_to_16k_mono(file_path, output_path):
            converted_files[filename_stem] = output_path
            print(f"✓ 成功轉換: {filename}")
        else:
            failed_files.append(filename)
            print(f"✗ 轉換失敗: {filename}")

    print(f"\n批次轉換完成！")
    print(f"成功轉換: {len(converted_files)} 個檔案")
    print(f"轉換失敗: {len(failed_files)} 個檔案")

    return converted_files, failed_files


def create_manifest_with_conversion(audio_dir, transcript_path, manifest_output_path,
                                    convert_audio=True, converted_audio_dir=None,
                                    normalize_text=True):
    """
    產生 manifest，可選擇是否先轉換音訊檔案為16kHz單聲道格式，並對文本進行正規化。
    所有路徑都使用絕對路徑。
    """
    manifest_entries = []
    missing_files_count = 0
    processed_files_count = 0
    converted_files = {}

    # 轉換為絕對路徑
    audio_dir_abs = os.path.abspath(audio_dir)
    transcript_path_abs = os.path.abspath(transcript_path)
    manifest_output_path_abs = os.path.abspath(manifest_output_path)

    print(f"開始處理目錄 '{audio_dir_abs}' 和文字稿 '{transcript_path_abs}'...")
    if normalize_text:
        print("文本正規化功能已啟用")

    if not os.path.isdir(audio_dir_abs):
        print(f"錯誤：音訊目錄 '{audio_dir_abs}' 不存在。")
        return
    if not os.path.isfile(transcript_path_abs):
        print(f"錯誤：文字稿檔案 '{transcript_path_abs}' 不存在。")
        return

    # 如果需要轉換音訊檔案
    if convert_audio:
        if converted_audio_dir is None:
            converted_audio_dir_abs = os.path.join(os.path.dirname(audio_dir_abs), "converted_audio_16k")
        else:
            converted_audio_dir_abs = os.path.abspath(converted_audio_dir)

        print(f"開始轉換音訊檔案到 '{converted_audio_dir_abs}'...")
        converted_files, failed_files = batch_convert_audio(audio_dir_abs, converted_audio_dir_abs)

        if failed_files:
            print(f"警告：以下檔案轉換失敗: {failed_files}")

        # 使用轉換後的音訊目錄
        working_audio_dir_abs = converted_audio_dir_abs
    else:
        working_audio_dir_abs = audio_dir_abs

    line_num = 0
    with open(transcript_path_abs, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t', 1)
            if len(parts) < 2:
                print(f"警告 (行 {line_num}): 跳過格式不正確的行: '{line}'")
                continue

            audio_filename_stem, text_content = parts

            # 對文本進行正規化處理
            if normalize_text:
                original_text = text_content
                text_content = normalizer(text_content)
                print(f"文本正規化 (行 {line_num}): '{original_text}' -> '{text_content}'")

            # 如果進行了轉換，使用轉換後的檔案（已經是絕對路徑）
            if convert_audio:
                if audio_filename_stem in converted_files:
                    audio_filepath_full = converted_files[audio_filename_stem]
                else:
                    print(f"警告 (行 {line_num}): 找不到轉換後的音訊檔案 '{audio_filename_stem}.wav'，跳過。")
                    missing_files_count += 1
                    continue
            else:
                audio_filepath_full = os.path.abspath(os.path.join(working_audio_dir_abs, f"{audio_filename_stem}.wav"))

                if not os.path.exists(audio_filepath_full):
                    print(f"警告 (行 {line_num}): 找不到音訊檔案 '{audio_filepath_full}'，跳過。")
                    missing_files_count += 1
                    continue

            try:
                # 讀取音訊檔案資訊以取得時長
                audio_info = sf.info(audio_filepath_full)
                duration = audio_info.duration

                # 驗證音訊格式（如果需要轉換的話）
                if convert_audio:
                    if audio_info.samplerate != 16000:
                        print(f"警告 (行 {line_num}): 音訊檔案 '{audio_filepath_full}' 採樣率不是16kHz")
                    if audio_info.channels != 1:
                        print(f"警告 (行 {line_num}): 音訊檔案 '{audio_filepath_full}' 不是單聲道")

            except Exception as e:
                print(f"警告 (行 {line_num}): 無法讀取音訊檔案 '{audio_filepath_full}': {e}，跳過。")
                missing_files_count += 1
                continue

            # 建立 manifest 條目（使用絕對路徑和正規化後的文本）
            manifest_entries.append({
                'audio_filepath': audio_filepath_full,
                'text': text_content,
                'duration': duration
            })
            processed_files_count += 1

    if not manifest_entries and processed_files_count == 0 and missing_files_count == 0:
        print(f"警告：文字稿檔案 '{transcript_path_abs}' 為空或所有行均無法處理。未產生任何 manifest 條目。")

    # 寫入 manifest 檔案
    try:
        os.makedirs(os.path.dirname(manifest_output_path_abs), exist_ok=True)
        with open(manifest_output_path_abs, 'w', encoding='utf-8') as f_manifest:
            for entry in manifest_entries:
                json.dump(entry, f_manifest, ensure_ascii=False)
                f_manifest.write('\n')
    except Exception as e_manifest:
        print(f"錯誤：無法寫入 manifest 檔案 '{manifest_output_path_abs}': {e_manifest}")
        return

    print(f"清單檔案已成功建立於: {manifest_output_path_abs}")
    print(f"總共處理文字稿行數: {line_num}")
    print(f"成功加入 manifest 的條目數: {processed_files_count}")
    print(f"因檔案缺失或處理錯誤而跳過的條目數: {missing_files_count}")

    if convert_audio:
        print(f"轉換後的音訊檔案儲存於: {working_audio_dir_abs}")


# --- 範例使用說明 ---
if __name__ == '__main__':
    # 設定是否要轉換音訊檔案和文本正規化
    CONVERT_AUDIO = True  # 設為 True 來啟用音訊轉換功能
    NORMALIZE_TEXT = True  # 設為 True 來啟用文本正規化功能

    # 使用絕對路徑
    TRAIN_AUDIO_DIR = os.path.abspath("./Test/audio_NoBGM")
    TRAIN_TRANSCRIPT_ORIGINAL = os.path.abspath("./train80/task1_answer.txt")
    TRAIN_MANIFEST_OUTPUT = os.path.abspath("./train80/train_manifest.json")
    TRAIN_CONVERTED_AUDIO_DIR = os.path.abspath("./train80/audio_16k")
    #
    VAL_AUDIO_DIR = os.path.abspath("./val20/audio_NoBGM")
    VAL_TRANSCRIPT_ORIGINAL = os.path.abspath("./val20/task1_answer.txt")
    VAL_MANIFEST_OUTPUT = os.path.abspath("./val20/val_manifest.json")
    VAL_CONVERTED_AUDIO_DIR = os.path.abspath("./val20/audio_16k")

    CH_AUDIO_DIR = os.path.abspath("./chinese/audio_noBGM")
    CH_TRANSCRIPT_ORIGINAL = os.path.abspath("./chinese/task1_answer.txt")
    CH_MANIFEST_OUTPUT = os.path.abspath("./chinese/manifest.json")
    CH_CONVERTED_AUDIO_DIR = os.path.abspath("chinese/audio_16k")

    # print("--- 開始產生訓練資料集的清單檔案 ---")
    create_manifest_with_conversion(
        TRAIN_AUDIO_DIR,
        TRAIN_TRANSCRIPT_ORIGINAL,
        TRAIN_MANIFEST_OUTPUT,
        convert_audio=CONVERT_AUDIO,
        converted_audio_dir=TRAIN_CONVERTED_AUDIO_DIR,
        normalize_text=NORMALIZE_TEXT
    )

    print("\n--- 開始產生驗證資料集的清單檔案 ---")
    create_manifest_with_conversion(
        VAL_AUDIO_DIR,
        VAL_TRANSCRIPT_ORIGINAL,
        VAL_MANIFEST_OUTPUT,
        convert_audio=CONVERT_AUDIO,
        converted_audio_dir=VAL_CONVERTED_AUDIO_DIR,
        normalize_text=NORMALIZE_TEXT
    )

    # print("\n--- 開始產生中文資料集的清單檔案 ---")
    create_manifest_with_conversion(
        CH_AUDIO_DIR,
        CH_TRANSCRIPT_ORIGINAL,
        CH_MANIFEST_OUTPUT,
        convert_audio=CONVERT_AUDIO,
        converted_audio_dir=CH_CONVERTED_AUDIO_DIR,
        normalize_text=NORMALIZE_TEXT
    )

    print("\n--- 所有清單檔案產生完畢 ---")
    print(f"請檢查 '{TRAIN_MANIFEST_OUTPUT}' 和 '{VAL_MANIFEST_OUTPUT}' 檔案。")
    if CONVERT_AUDIO:
        print(f"轉換後的音訊檔案位於:")
        print(f"  - 訓練資料: {TRAIN_CONVERTED_AUDIO_DIR}")
        print(f"  - 驗證資料: {VAL_CONVERTED_AUDIO_DIR}")
    if NORMALIZE_TEXT:
        print("文本已使用Whisper EnglishTextNormalizer進行正規化處理")
    print("如果沒有錯誤，您可以在您的 NeMo 微調腳本中使用這些新的 .json 清單檔案。")
