import os
import re


def rename_vocal_wav_files(folder_path):
    """
    重新命名指定資料夾下，符合特定命名模式的 .wav 檔案。

    將檔案名稱從 `{id}_{origName}_(Vocals).wav` 轉換回 `{origName}.wav`。

    Args:
        folder_path (str): 包含 .wav 檔案的資料夾路徑。
    """
    print(f"正在檢查資料夾：'{folder_path}' 中的 .wav 檔案...")

    # 正則表達式用於匹配 `{id}_{origName}_(Vocals).wav` 格式
    # id 可以是數字、字母或底線的組合
    # origName 可以是任何字元，但不能包含底線後緊接著 (Vocals)
    # 這裡假設 origName 不會包含 `_(Vocals)` 這種字串
    pattern = re.compile(r"(.+?)_(.+?)_\(Vocals\)\.wav$", re.IGNORECASE)

    renamed_count = 0
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".wav"):
            match = pattern.match(filename)
            if match:
                # 取得 origName，這裡 match.group(2) 包含了原始名稱
                # match.group(1) 是 id
                # match.group(2) 是 origName
                orig_name = match.group(2)
                new_filename = f"{orig_name}.wav"
                old_filepath = os.path.join(folder_path, filename)
                new_filepath = os.path.join(folder_path, new_filename)

                try:
                    os.rename(old_filepath, new_filepath)
                    print(f"已將 '{filename}' 重新命名為 '{new_filename}'")
                    renamed_count += 1
                except OSError as e:
                    print(f"重新命名 '{filename}' 失敗：{e}")
            # else:
            #     print(f"檔案 '{filename}' 不符合重新命名模式，跳過。")
    if renamed_count == 0:
        print("沒有找到符合重新命名模式的檔案。")
    else:
        print(f"總共重新命名了 {renamed_count} 個檔案。")


# ---
# 範例使用方式：
# 請將 'your_folder_path_here' 替換成你要處理的資料夾路徑。
# 例如：
# rename_vocal_wav_files("C:/Users/YourUser/Music/VocalSeparation")
# 或者對於相對路徑：
# rename_vocal_wav_files("./my_audio_files")

if __name__ == '__main__':
    rename_vocal_wav_files("./train80/audio_NoBGM")
    rename_vocal_wav_files("./val20/audio_NoBGM")
    rename_vocal_wav_files("./test/audio_NoBGM")
