import os
import subprocess
import shutil
import argparse

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
        print(f"轉換音訊檔案 '{input_file}' 時發生錯誤: {e}")
        return False

def batch_convert_audio(input_dir, output_dir, supported_formats=None):
    """
    批次轉換資料夾中的音訊檔案為16kHz單聲道WAV格式
    """
    if supported_formats is None:
        supported_formats = ['.mp3', '.m4a', '.aac', '.flac', '.wav', '.ogg', '.wma']

    converted_count = 0
    failed_files = []

    # 轉換為絕對路徑
    input_dir_abs = os.path.abspath(input_dir)
    output_dir_abs = os.path.abspath(output_dir)

    print(f"開始批次轉換音訊檔案從 '{input_dir_abs}' 到 '{output_dir_abs}'...")

    if not os.path.exists(input_dir_abs):
        print(f"錯誤：輸入目錄 '{input_dir_abs}' 不存在。")
        return

    if not shutil.which('ffmpeg'):
        print("錯誤：找不到FFmpeg。請確保已安裝FFmpeg並添加到系統PATH中。")
        return

    os.makedirs(output_dir_abs, exist_ok=True)

    # 找出所有支援的音訊檔案
    files_to_process = [
        f for f in os.listdir(input_dir_abs) 
        if os.path.splitext(f)[1].lower() in supported_formats and os.path.isfile(os.path.join(input_dir_abs, f))
    ]

    if not files_to_process:
        print("在輸入目錄中找不到支援格式的音訊檔案。")
        return

    print(f"找到 {len(files_to_process)} 個要處理的檔案。")

    for filename in files_to_process:
        file_path = os.path.join(input_dir_abs, filename)
        
        # 產生輸出檔案路徑
        filename_stem = os.path.splitext(filename)[0]
        output_filename = f"{filename_stem}.wav"
        output_path = os.path.join(output_dir_abs, output_filename)

        print(f"正在轉換: {filename} -> {output_filename}")

        if convert_audio_to_16k_mono(file_path, output_path):
            converted_count += 1
            print(f"✓ 成功轉換: {filename}")
        else:
            failed_files.append(filename)
            print(f"✗ 轉換失敗: {filename}")

    print("\n批次轉換完成！")
    print(f"成功轉換: {converted_count} 個檔案")
    print(f"轉換失敗: {len(failed_files)} 個檔案")
    if failed_files:
        print("失敗的檔案列表:")
        for f in failed_files:
            print(f"  - {f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="批次轉換音訊檔案為 16kHz 單聲道 WAV 格式。",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
範例用法:
  python %(prog)s ./my_audio ./converted_audio
"""
    )
    parser.add_argument("--input_dir", help="包含原始音訊檔案的輸入目錄。")
    parser.add_argument("--output_dir", help="儲存轉換後 .wav 檔案的輸出目錄。")
    
    args = parser.parse_args()
    
    batch_convert_audio(args.input_dir, args.output_dir) 