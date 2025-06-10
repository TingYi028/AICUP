import os
import argparse
import nemo.collections.asr as nemo_asr
import librosa
import soundfile as sf
from typing import List, Optional, Dict, Any
from jiwer import mer, wer
from transformers.models.whisper.english_normalizer import BasicTextNormalizer, EnglishTextNormalizer
from tqdm import tqdm
import re
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def calculate_mer(ground_truth_texts, predicted_texts):
    """ Mix Error Rate (MER) English only"""
    mer_scores = {}
    total_mer = 0
    count = 0

    normalizer = EnglishTextNormalizer({})

    for filename, ref_text in ground_truth_texts.items():
        if filename in predicted_texts:
            pred_text = predicted_texts[filename]
            ref_text = normalizer(ref_text)
            pred_text = normalizer(pred_text)

            mer_score = mer(ref_text, pred_text)
            mer_scores[filename] = mer_score
            total_mer += mer_score
        else:
            mer_scores[filename] = 1
            total_mer += 1
        count += 1

    average_mer = total_mer / count if count != 0 else 0
    return mer_scores, average_mer


class ParakeetASRPredictor:
    """NVIDIA Parakeet TDT 0.6B V2 語音識別預測器 - 批量處理版"""

    def __init__(self, model_name: str = "nvidia/parakeet-tdt-0.6b-v2"):
        """
        初始化 ASR 模型

        Args:
            model_name: 模型名稱，預設為 nvidia/parakeet-tdt-0.6b-v2
        """
        model_name = "nvidia/parakeet-tdt-0.6b-v2"
        print(f"🚀 正在載入模型: {model_name}")
        print("⏳ 模型載入中，請稍候...")
        self.asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name)
        # self.asr_model = nemo_asr.models.ASRModel.restore_from(model_name)

        print("✅ 模型載入完成！")

    def preprocess_audio(self, audio_path: str, target_sr: int = 16000) -> str:
        """
        預處理音頻文件，確保符合模型要求

        Args:
            audio_path: 音頻文件路徑
            target_sr: 目標採樣率 (16kHz)

        Returns:
            處理後的音頻文件路徑
        """
        # 讀取音頻
        audio, sr = librosa.load(audio_path, sr=None)

        # 檢查是否需要處理
        needs_processing = False

        # 轉換為單聲道
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
            needs_processing = True

        # 重新採樣到 16kHz
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            needs_processing = True

        # 如果不需要處理，直接返回原文件
        if not needs_processing:
            return audio_path

        # 正確處理文件路徑
        base_path, ext = os.path.splitext(audio_path)
        processed_path = f"{base_path}_processed{ext}"

        # 確保目錄存在
        processed_dir = os.path.dirname(processed_path)
        if processed_dir:
            os.makedirs(processed_dir, exist_ok=True)

        sf.write(processed_path, audio, target_sr)

        return processed_path

    def load_labels(self, label_file: str) -> Dict[str, str]:
        """
        讀取標籤文件

        Args:
            label_file: 標籤文件路徑

        Returns:
            文件名到文本的映射字典
        """
        print(f"📖 正在讀取標籤文件: {label_file}")
        labels = {}
        with open(label_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in tqdm(lines, desc="讀取標籤", unit="行"):
            line = line.strip()
            if line:
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    file_id, text = parts
                    # 假設音頻文件名格式為 {file_id}.wav
                    labels[f"{file_id}.wav"] = text

        print(f"✅ 成功讀取 {len(labels)} 個標籤")
        return labels

    def batch_transcribe(self, audio_dir: str, output_file: Optional[str] = None,
                         label_file: Optional[str] = None, output_label_file: str = "task1_answer_Finetuned.txt",
                         json_output_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        批量轉錄目錄中的音頻文件

        Args:
            audio_dir: 音頻文件目錄
            output_file: 輸出文件路徑（可選）
            label_file: 標籤文件路徑（可選，用於計算MER）
            output_label_file: 輸出標籤文件路徑（預設為output_label.txt）
            json_output_file: JSON時間戳輸出文件路徑（可選）

        Returns:
            轉錄結果列表
        """
        print(f"🔍 正在掃描音頻目錄: {audio_dir}")

        # 支援的音頻格式
        supported_formats = ['.wav', '.flac', '.mp3', '.m4a']

        # 獲取所有音頻文件
        audio_files = []
        all_files = os.listdir(audio_dir)

        for file in tqdm(all_files, desc="掃描文件", unit="個"):
            if any(file.lower().endswith(fmt) for fmt in supported_formats):
                audio_files.append(os.path.join(audio_dir, file))

        if not audio_files:
            print(f"❌ 在目錄 {audio_dir} 中未找到支援的音頻文件")
            print(f"支援格式: {', '.join(supported_formats)}")
            return []

        print(f"✅ 找到 {len(audio_files)} 個音頻文件")

        # 預處理音頻文件
        print("🔧 正在預處理音頻文件...")
        processed_files = []
        failed_files = []

        for audio_file in tqdm(audio_files, desc="預處理音頻", unit="個"):
            try:
                processed_file = self.preprocess_audio(audio_file)
                processed_files.append(processed_file)
            except Exception as e:
                print(f"\n⚠️  處理文件 {os.path.basename(audio_file)} 時出錯: {e}")
                failed_files.append(audio_file)

        if failed_files:
            print(f"⚠️  {len(failed_files)} 個文件處理失敗")

        if not processed_files:
            print("❌ 沒有成功處理的音頻文件")
            return []

        print(f"✅ 成功預處理 {len(processed_files)} 個音頻文件")

        # 批量轉錄
        print("🎤 開始批量轉錄...")
        print("⏳ 正在進行語音識別，請稍候...")
        outputs = self.asr_model.transcribe(processed_files, timestamps=True, batch_size=4)
        print("✅ 轉錄完成！")

        # 整理結果
        print("📊 正在整理轉錄結果...")
        final_results = []

        # 需要調整索引，因為可能有失敗的文件
        successful_audio_files = [f for f in audio_files if f not in failed_files]

        # 初始化英文文本正規化器
        normalizer = EnglishTextNormalizer({})

        for i, output in enumerate(tqdm(outputs, desc="整理結果", unit="個")):
            # 使用 Whisper 的英文正規化器處理文本
            converted_text = normalizer(output.text)

            result = {
                'file': successful_audio_files[i],
                'transcription': converted_text,
                'original_transcription': output.text,  # 保留原始轉錄結果
                'word_timestamps': output.timestamp.get('word', []),
                'segment_timestamps': output.timestamp.get('segment', []),
                'char_timestamps': output.timestamp.get('char', [])
            }
            final_results.append(result)

        # **新增：輸出 task1_answer_Finetuned.txt 文件**
        print(f"📄 正在輸出標籤文件: {output_label_file}")
        os.makedirs(os.path.dirname(output_label_file), exist_ok=True)
        with open(output_label_file, 'w', encoding='utf-8') as f:
            for result in tqdm(final_results, desc="寫入標籤文件", unit="個"):
                filename = os.path.basename(result['file'])
                # 移除副檔名，只保留檔名
                filename_without_ext = os.path.splitext(filename)[0]
                transcription = result['transcription']
                f.write(f"{filename_without_ext}\t{transcription}\n")
        print(f"✅ 標籤文件已成功保存到: {output_label_file}")

        # **修改：輸出新格式的JSON時間戳文件**
        if json_output_file:
            print(f"📄 正在輸出JSON時間戳文件: {json_output_file}")

            json_data = {}

            for result in tqdm(final_results, desc="生成JSON時間戳", unit="個"):
                filename = os.path.basename(result['file'])
                # 移除副檔名，只保留檔名
                filename_without_ext = os.path.splitext(filename)[0]

                # 提取詞級別時間戳
                word_timestamps = result['word_timestamps']
                timestamp_list = []

                for word_stamp in word_timestamps:
                    timestamp_entry = {
                        "startTime": word_stamp['start'],
                        "endTime": word_stamp['end'],
                        "Text": word_stamp['word']
                    }
                    timestamp_list.append(timestamp_entry)

                # **新格式：包含transcription和wordTimeStamp**
                json_data[filename_without_ext] = {
                    "transcription": result['transcription'],
                    "wordTimeStamp": timestamp_list
                }

            # 寫入JSON檔案
            with open(json_output_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)

            print(f"✅ JSON時間戳文件已成功保存到: {json_output_file}")

        # 計算MER指標（如果提供了標籤文件）
        mer_scores = None
        average_mer = None
        normalized_ground_truth = {}  # 新增：儲存正規化後的標籤

        if label_file:
            print("📈 正在計算MER指標...")
            ground_truth = self.load_labels(label_file)
            predicted_texts = {}

            for result in tqdm(final_results, desc="準備MER計算", unit="個"):
                filename = os.path.basename(result['file'])
                predicted_texts[filename] = result['transcription']

            print("🧮 計算MER分數中...")
            mer_scores, average_mer = calculate_mer(ground_truth, predicted_texts)
            print(f"✅ 平均MER: {average_mer:.4f}")

            # 正規化標籤文本並儲存
            for filename, original_label in ground_truth.items():
                normalized_ground_truth[filename] = normalizer(original_label)

        # 保存詳細結果到文件
        if output_file:
            print(f"💾 正在保存詳細結果到: {output_file}")
            with open(output_file, 'w', encoding='utf-8') as f:
                # 寫入MER結果（如果有）
                if mer_scores is not None:
                    f.write(f"平均MER: {average_mer:.4f}\n")
                    f.write("=" * 50 + "\n")
                    f.write("各文件MER分數:\n")
                    for filename, score in mer_scores.items():
                        f.write(f"{filename}: {score:.4f}\n")
                    f.write("=" * 50 + "\n\n")

                # 寫入轉錄結果
                for result in tqdm(final_results, desc="寫入詳細結果", unit="個"):
                    filename = os.path.basename(result['file'])
                    f.write(f"文件: {result['file']}\n")

                    # 如果有標籤文件，寫入標籤文本
                    if label_file and mer_scores is not None:
                        # ground_truth 已經在計算MER時載入過了
                        if filename in ground_truth:
                            f.write(f"標籤文本: {ground_truth[filename]}\n")
                            f.write(f"正規化標籤: {normalized_ground_truth[filename]}\n")
                        else:
                            f.write(f"標籤文本: [未找到對應標籤]\n")

                    f.write(f"原始轉錄: {result['original_transcription']}\n")
                    f.write(f"正規化後: {result['transcription']}\n")

                    # 如果有MER分數，也寫入
                    if mer_scores and filename in mer_scores:
                        f.write(f"MER: {mer_scores[filename]:.4f}\n")

                    f.write("段落時間戳:\n")
                    for stamp in result['segment_timestamps']:
                        f.write(f"  {stamp['start']:.2f}s - {stamp['end']:.2f}s : {stamp['segment']}\n")
                    f.write("詞時間戳:\n")
                    for stamp in result['word_timestamps']:
                        f.write(f"  {stamp['start']:.2f}s - {stamp['end']:.2f}s : {stamp['word']}\n")
                    f.write("\n" + "=" * 50 + "\n\n")
            print(f"✅ 詳細結果已成功保存到: {output_file}")

        # 清理臨時文件
        print("🧹 正在清理臨時文件...")
        cleaned_count = 0
        for i, processed_file in enumerate(processed_files):
            if processed_file != successful_audio_files[i]:
                try:
                    os.remove(processed_file)
                    cleaned_count += 1
                except:
                    pass

        if cleaned_count > 0:
            print(f"✅ 已清理 {cleaned_count} 個臨時文件")

        # 將MER結果加入返回值
        if mer_scores is not None:
            return final_results, mer_scores, average_mer
        else:
            return final_results


def main():
    parser = argparse.ArgumentParser(description='NVIDIA Parakeet TDT 0.6B V2 批量語音識別')
    parser.add_argument('--audio_dir',default="../datasets/Test/audio_NoBGM_16k", type=str, help='音頻文件目錄路徑')
    parser.add_argument('--output', type=str, help='詳細輸出文件路徑')
    parser.add_argument('--labels',default="../datasets/val/task1_answer.txt", type=str, help='標籤文件路徑（用於計算MER）')
    parser.add_argument('--output_label', type=str, default='Test_result/task1_answer_noFinetuned.txt',
                        help='輸出標籤文件路徑（預設為output_label.txt）')
    parser.add_argument('--json_output',default="Test_result/task1_answer_noFinetuned_timeStamp.json", type=str, help='JSON時間戳輸出文件路徑')
    parser.add_argument('--model', type=str, default=r'C:\Users\C110151154\PycharmProjects\AICUP\AICUP\ASR\fintuned_model\Speech_To_Text_Finetuning.nemo', help='模型名稱')

    args = parser.parse_args()
    # 初始化預測器
    predictor = ParakeetASRPredictor(model_name=args.model)

    # 批量轉錄
    print(f"🎯 開始批量轉錄任務")
    print(f"📁 目標目錄: {args.audio_dir}")
    if args.output:
        print(f"📄 詳細輸出文件: {args.output}")
    print(f"🏷️  標籤輸出文件: {args.output_label}")
    if args.labels:
        print(f"🏷️  參考標籤文件: {args.labels}")
    if args.json_output:
        print(f"📊 JSON時間戳文件: {args.json_output}")
    print("-" * 50)

    result = predictor.batch_transcribe(
        args.audio_dir,
        args.output,
        args.labels,
        args.output_label,
        args.json_output
    )

    # 處理返回結果
    print("\n" + "=" * 60)
    print("🎉 批量轉錄任務完成！")
    print("=" * 60)
    args.labels = None
    if args.labels:
        results, mer_scores, average_mer = result
        # 載入標籤文件用於顯示
        ground_truth = predictor.load_labels(args.labels)

        print(f"📊 處理統計:")
        print(f"   ✅ 成功處理: {len(results)} 個文件")
        print(f"   📈 平均MER: {average_mer:.4f}")
        print(f"   📄 標籤文件: {args.output_label}")
        if args.json_output:
            print(f"   📊 JSON時間戳: {args.json_output}")

        print(f"\n📝 轉錄結果預覽 (前5個文件):")
        print("-" * 50)
        for i, result_item in enumerate(results[:5]):
            filename = os.path.basename(result_item['file'])
            print(f"🎵 {i + 1}. {filename}")

            # 如果有標籤，顯示標籤文本
            if filename in ground_truth:
                print(f"   🏷️  標籤: {ground_truth[filename][:80]}...")

            print(f"   📝 原始: {result_item['original_transcription'][:80]}...")
            print(f"   🔢 正規化: {result_item['transcription'][:80]}...")
            if filename in mer_scores:
                print(f"   📊 MER: {mer_scores[filename]:.4f}")
            print()
    else:
        results = result
        print(f"📊 處理統計:")
        print(f"   ✅ 成功處理: {len(results)} 個文件")
        print(f"   📄 標籤文件: {args.output_label}")
        if args.json_output:
            print(f"   📊 JSON時間戳: {args.json_output}")

        print(f"\n📝 轉錄結果預覽 (前5個文件):")
        print("-" * 50)
        for i, result_item in enumerate(results[:5]):
            filename = os.path.basename(result_item['file'])
            print(f"🎵 {i + 1}. {filename}")
            print(f"   📝 原始: {result_item['original_transcription'][:80]}...")
            print(f"   🔢 正規化: {result_item['transcription'][:80]}...")
            print()

    if len(results) > 5:
        print(f"... 還有 {len(results) - 5} 個文件的結果")

    print("🎊 任務完成！")


if __name__ == "__main__":
    main()
