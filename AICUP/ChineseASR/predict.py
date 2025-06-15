import os
import torch
import jiwer
import json
from opencc import OpenCC  # <<< 新增：導入 OpenCC
from tqdm import tqdm
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer, EnglishTextNormalizer  # 用於計算 MER

normalizer = BasicTextNormalizer()
# --- 配置參數 ---
# !!! 重要：請修改為您微調模型的實際路徑 (通常是訓練腳本中的 OUTPUT_DIR) !!!
MODEL_NAME = "openai/whisper-large-v3-turbo"  # 處理器可以從基礎模型載入，確保兼容性

VAL_AUDIO_DATA_DIR = r"../datasets/test/zh"
# VAL_LABEL_FILE_PATH = r"C:\Users\C110151154\PycharmProjects\NeMo\AICUP\datasets\TestChinese\task1_answer.txt"
VAL_LABEL_FILE_PATH = None
OUTPUT_FILE_PATH = "test_result/task1_answer_zh.txt"  # 原始純文本輸出
OUTPUT_JSON_FILE_PATH = "test_result/task1_answer_zh.json"  # 新增的 JSON 輸出檔案
AUDIO_EXTENSION = ".wav"

GENERATION_MAX_LENGTH = 250  # 建議與訓練時的設定保持一致或按需調整


# --- 1. 資料載入函數 ---
def load_audio_paths_and_labels(audio_dir, label_file_path, audio_ext=".wav"):
    """
    載入音訊檔案路徑、真實標籤 (如果可用) 和檔名。
    如果提供了 label_file_path 且存在，則從中讀取文件名和真實標籤。
    否則，掃描 audio_dir 中的所有音訊檔案。
    返回:
        - audio_file_paths (list): 音訊檔案的完整路徑列表。
        - ground_truths (list): 真實標籤列表 (如果不可用則為空字串)。
        - file_names_stem (list): 音訊檔案名列表 (不含副檔名)，順序與 audio_file_paths 一致。
    """
    audio_file_paths = []
    ground_truths = []
    file_names_stem = []
    processed_stems = set()

    if label_file_path and os.path.exists(label_file_path):
        print(f"從標籤檔案讀取音訊列表和真實標籤: {label_file_path}")
        with open(label_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    file_stem, text = line.split("\t", 1)
                    if file_stem in processed_stems:
                        continue
                    audio_path = os.path.join(audio_dir, file_stem + audio_ext)
                    if os.path.exists(audio_path):
                        audio_file_paths.append(audio_path)
                        ground_truths.append(text)
                        file_names_stem.append(file_stem)
                        processed_stems.add(file_stem)
                    else:
                        print(f"警告: 標籤檔案中指定的音訊檔案 {audio_path} 未找到，已跳過。")
                except ValueError:
                    print(f"警告: 無法解析標籤檔案中的行 '{line}'，格式應為 'fileName\\ttext'。已跳過。")
    else:
        print(f"提示: 標籤檔案 '{label_file_path}' 未提供或不存在。")
        print(f"將掃描音訊目錄 '{audio_dir}' 中的所有 '{audio_ext}' 檔案進行預測。")
        if not os.path.exists(audio_dir):
            print(f"錯誤: 音訊目錄 '{audio_dir}' 不存在。")
            return [], [], []

        for f_name in sorted(os.listdir(audio_dir)):
            if f_name.endswith(audio_ext):
                file_stem = os.path.splitext(f_name)[0]
                if file_stem in processed_stems:
                    continue
                audio_file_paths.append(os.path.join(audio_dir, f_name))
                ground_truths.append("")  # 沒有真實標籤，使用空字串
                file_names_stem.append(file_stem)
                processed_stems.add(file_stem)

    if not audio_file_paths:
        print(f"警告: 在目錄 {audio_dir} 中找不到任何音訊檔案 (或根據標籤檔案)。無法進行預測。")
        return [], [], []

    print(f"成功準備 {len(audio_file_paths)} 個音訊檔案進行預測。")
    return audio_file_paths, ground_truths, file_names_stem


# --- 2. 評估指標計算函數 ---
def compute_mer_for_pipeline(ground_truths_list, predictions_list, normalizer):
    """計算 MER，輸入為正規化後的真實標籤和預測文字列表。"""
    normalized_label_str = [normalizer(s) for s in ground_truths_list]
    normalized_pred_str = [normalizer(s) for s in predictions_list]

    if not normalized_label_str and not normalized_pred_str:
        return 0.0
    if all(not s.strip() for s in normalized_label_str) and not any(p.strip() for p in normalized_pred_str):
        return 0.0

    filtered_labels = []
    filtered_preds = []
    for lbl, pred_txt in zip(normalized_label_str, normalized_pred_str):
        if lbl.strip():
            filtered_labels.append(lbl)
            filtered_preds.append(pred_txt)
        elif not lbl.strip() and not pred_txt.strip():
            filtered_labels.append(lbl)
            filtered_preds.append(pred_txt)

    if not filtered_labels:
        if all(not s.strip() for s in normalized_label_str) and any(p.strip() for p in normalized_pred_str):
            return 1.0
        return 0.0

    try:
        mer = jiwer.mer(filtered_labels, filtered_preds)
    except Exception as e:
        print(f"計算 MER 時發生錯誤: {e}")
        print(f"Filtered Labels (前5): {filtered_labels[:5]}")
        print(f"Filtered Preds (前5): {filtered_preds[:5]}")
        mer = float('inf')
    return mer


# --- 3. 主預測邏輯 (使用 Pipeline) ---
def run_prediction_with_pipeline():
    """執行主要的預測流程 (使用 Hugging Face Pipeline)。"""
    global normalizer
    # <<< 新增：初始化 OpenCC 轉換器 (s2twp: 簡體到繁體台灣用語)
    cc = OpenCC('s2twp')

    # --- 設備和型態設定 ---
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    if not torch.cuda.is_available():
        print("警告: CUDA 不可用。將使用 CPU 進行預測，這可能會非常慢。")

    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch_dtype,
        )
        model.to(device)
        print(f"模型已移至設備: {device}")
    except Exception as e:
        print(f"錯誤：無法載入模型於 '{MODEL_NAME}'。")
        print(f"詳細錯誤: {e}")
        exit()

    # --- 載入處理器 ---
    print(f"載入處理器: {MODEL_NAME}")
    try:
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"錯誤：無法載入處理器於 '{MODEL_NAME}'。")
        print(f"詳細錯誤: {e}")
        exit()

    # --- 初始化 ASR Pipeline ---
    print("初始化 ASR pipeline...")
    try:
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
            chunk_length_s=30,
        )
    except Exception as e:
        print(f"錯誤：初始化 ASR pipeline 失敗。")
        print(f"詳細錯誤: {e}")
        exit()

    # --- 載入驗證資料 ---
    print("載入並準備驗證資料集進行預測...")
    actual_label_file_path = VAL_LABEL_FILE_PATH if VAL_LABEL_FILE_PATH and os.path.exists(
        VAL_LABEL_FILE_PATH) else None
    audio_file_paths, ground_truths, file_names_stem = load_audio_paths_and_labels(
        VAL_AUDIO_DATA_DIR,
        actual_label_file_path,
        audio_ext=AUDIO_EXTENSION
    )

    if not audio_file_paths:
        print("無法載入音訊檔案或文件名列表為空，終止預測。")
        return

    print(f"待預測的音訊數量: {len(audio_file_paths)}")
    has_ground_truth_labels = any(s.strip() for s in ground_truths) if actual_label_file_path else False

    # --- 執行預測 ---
    print(f"開始進行預測 (逐個檔案處理，總數: {len(audio_file_paths)})...")

    # 設置生成參數
    generate_kwargs = {
        "language": "zh",  # 指定語言，提高準確性
        "task": "transcribe",  # 指定任務
        "max_new_tokens": 300,
        "return_timestamps": True,
    }

    predicted_transcriptions_plain = []  # 用於純文本輸出
    predicted_json_data = {}  # 用於 JSON 輸出

    # 使用 tqdm 迭代 audio_file_paths 以顯示進度條
    for i, audio_path in enumerate(tqdm(audio_file_paths, desc="ASR 進度")):
        file_name_stem = file_names_stem[i]  # 獲取對應的檔名
        try:
            # 調用 pipeline 並要求 word-level timestamps
            result = asr_pipeline(
                audio_path,
                generate_kwargs=generate_kwargs,
                return_timestamps="word"
            )
            print(result)

            # <<< 修改：將完整的辨識結果轉換為繁體中文
            full_transcription = cc.convert(result["text"])
            predicted_transcriptions_plain.append(full_transcription)

            word_time_stamps = []
            if "chunks" in result and result["chunks"]:
                for chunk in result["chunks"]:
                    start_time = float(chunk["timestamp"][0])
                    end_time = float(chunk["timestamp"][1])

                    # <<< 修改：將 chunk 的文字正規化並轉換為繁體中文
                    text = cc.convert(normalizer(chunk["text"]).strip())

                    # 檢查文字是否可以用空格拆分
                    words = text.split()

                    if len(words) > 1:
                        # 計算每個單詞的平均時間長度
                        total_duration = end_time - start_time
                        time_per_word = total_duration / len(words)

                        # 為每個單詞分配時間戳記
                        for j, word in enumerate(words):
                            word_start_time = start_time + (j * time_per_word)
                            word_end_time = start_time + ((j + 1) * time_per_word)

                            word_time_stamps.append({
                                "startTime": round(word_start_time, 2),  # 精確到小數點後兩位
                                "endTime": round(word_end_time, 2),
                                "Text": word.strip()  # 清除單詞前後空格
                            })
                    elif text:  # 確保文字不為空
                        # 如果只有一個單詞或無法拆分，直接使用原始時間戳記
                        word_time_stamps.append({
                            "startTime": round(start_time, 2),
                            "endTime": round(end_time, 2),
                            "Text": text
                        })

            predicted_json_data[file_name_stem] = {
                "transcription": full_transcription,
                "wordTimeStamp": word_time_stamps
            }

        except Exception as e:
            file_basename = os.path.basename(audio_path)
            print(f"\n處理檔案 {file_basename} 時發生錯誤: {e}")
            predicted_transcriptions_plain.append(f"[錯誤處理檔案: {file_basename}]")
            predicted_json_data[file_name_stem] = {
                "transcription": f"[錯誤處理檔案: {file_basename}]",
                "wordTimeStamp": []
            }

    # --- 儲存純文本預測結果 ---
    print(f"\n將純文本預測結果寫入到檔案: {OUTPUT_FILE_PATH}")
    try:
        # 確保輸出目錄存在
        os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
        with open(OUTPUT_FILE_PATH, "w", encoding="utf-8") as f_out:
            for file_name, transcription in zip(file_names_stem, predicted_transcriptions_plain):
                clean_transcription = transcription.strip() if transcription else ""
                f_out.write(f"{file_name}\t{clean_transcription}\n")
        print(f"純文本預測結果已成功儲存到 {OUTPUT_FILE_PATH}")
    except IOError as e:
        print(f"寫入輸出檔案 {OUTPUT_FILE_PATH} 時發生錯誤: {e}")
        return

    # --- 儲存 JSON 預測結果 ---
    print(f"將帶時間戳記的預測結果寫入到檔案: {OUTPUT_JSON_FILE_PATH}")
    try:
        # 確保輸出目錄存在
        os.makedirs(os.path.dirname(OUTPUT_JSON_FILE_PATH), exist_ok=True)
        with open(OUTPUT_JSON_FILE_PATH, "w", encoding="utf-8") as f_json_out:
            json.dump(predicted_json_data, f_json_out, ensure_ascii=False, indent=2)
        print(f"JSON 預測結果已成功儲存到 {OUTPUT_JSON_FILE_PATH}")
    except IOError as e:
        print(f"寫入輸出檔案 {OUTPUT_JSON_FILE_PATH} 時發生錯誤: {e}")
        return

    # --- 計算並輸出評估指標 (如果可用) ---
    if has_ground_truth_labels:
        print("\n計算評估指標 (MER)...")
        # 僅當預測數量與真實標籤數量一致時才進行可靠的MER計算
        if len(predicted_transcriptions_plain) == len(ground_truths):
            try:
                # EnglishTextNormalizer 的初始化不需要詞彙表，或者可以傳遞一個空的字典
                mer_normalizer = EnglishTextNormalizer({})
                mer = compute_mer_for_pipeline(ground_truths, predicted_transcriptions_plain, mer_normalizer)
                print(f"  驗證集上的 MER: {mer:.4f}")
            except Exception as e:
                print(f"計算 MER 時發生錯誤: {e}")
                print("將嘗試不帶正規化地打印原始轉錄以供參考（如果數量匹配）：")
                if len(ground_truths) == len(predicted_transcriptions_plain):
                    print(f"  真實標籤 (前5項，未正規化): {ground_truths[:5]}")
                    print(f"  預測 (前5項，未正規化): {predicted_transcriptions_plain[:5]}")
        else:
            print("由於預測結果數量與真實標籤數量不符（可能因處理錯誤導致），跳過 MER 計算。")
    else:
        print("\n未提供真實標籤 (或標籤文件未找到)，跳過評估指標計算。")


if __name__ == "__main__":
    run_prediction_with_pipeline()
    print("\n使用 Pipeline 的預測腳本執行完畢！")