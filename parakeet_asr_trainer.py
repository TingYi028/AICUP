import os
import lightning.pytorch as pl
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf, open_dict

# --- 使用由輔助腳本產生的 JSON 清單檔案路徑 ---
TRAIN_MANIFEST = "train_manifest.json"
VAL_MANIFEST = "val_manifest.json"

# 檢查清單檔案是否存在
if not os.path.exists(TRAIN_MANIFEST):
    raise FileNotFoundError(f"訓練清單檔案未找到: {TRAIN_MANIFEST}. 請先執行 generate_manifest.py 並確保 'answer' 為文字稿鍵名。")
if not os.path.exists(VAL_MANIFEST):
    raise FileNotFoundError(f"驗證清單檔案未找到: {VAL_MANIFEST}. 請先執行 generate_manifest.py 並確保 'answer' 為文字稿鍵名。")

# 載入預訓練模型
print("正在載入預訓練模型 nvidia/parakeet-tdt-0.6b-v2...")
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
print("模型載入完畢。")

# 取得模型的設定物件
cfg = asr_model.cfg

# --- 設定訓練資料集 ---
print("正在設定訓練資料集...")
with open_dict(cfg.train_ds):
    cfg.train_ds.manifest_filepath = TRAIN_MANIFEST
    if hasattr(cfg.train_ds, 'use_lhotse'):
        cfg.train_ds.use_lhotse = False
    if hasattr(cfg.train_ds, 'use_bucketing'):
        cfg.train_ds.use_bucketing = False
    cfg.train_ds.batch_size = 2 # 根據您的 GPU 記憶體調整
    cfg.train_ds.shuffle = True
    # ***修正點：將 num_workers 設定為 0 以避免 PicklingError***
    cfg.train_ds.num_workers = 0
    cfg.train_ds.pin_memory = False # 當 num_workers 為 0 時，pin_memory 通常也應為 False 或可能引發警告
    if hasattr(cfg.train_ds, 'batch_duration'):
        cfg.train_ds.batch_duration = None
    if hasattr(cfg.train_ds, 'bucketing_batch_size'):
        cfg.train_ds.bucketing_batch_size = None
print("訓練資料集設定完畢。")

# --- 設定驗證資料集 ---
print("正在設定驗證資料集...")
with open_dict(cfg.validation_ds):
    cfg.validation_ds.manifest_filepath = VAL_MANIFEST
    if hasattr(cfg.validation_ds, 'use_lhotse'):
        cfg.validation_ds.use_lhotse = False
    if hasattr(cfg.validation_ds, 'use_bucketing'):
        cfg.validation_ds.use_bucketing = False
    cfg.validation_ds.batch_size = 2 # 根據您的 GPU 記憶體調整
    cfg.validation_ds.shuffle = False
    # ***修正點：將 num_workers 設定為 0 以避免 PicklingError***
    cfg.validation_ds.num_workers = 0
    cfg.validation_ds.pin_memory = False # 當 num_workers 為 0 時，pin_memory 通常也應為 False
print("驗證資料集設定完畢。")

# 將修改後的組態應用到模型
print("正在將資料載入器設定套用到模型...")
asr_model.setup_training_data(train_data_config=cfg.train_ds)
asr_model.setup_validation_data(val_data_config=cfg.validation_ds)
print("資料載入器設定完畢。")

# 設定 PyTorch Lightning Trainer
print("正在設定 PyTorch Lightning Trainer...")
# 建議在腳本開頭設定，以利用 Tensor Cores
if pl.accelerators.cuda.CUDAAccelerator.is_available():
    import torch
    torch.set_float32_matmul_precision('medium') # 或 'high'
    accelerator = "gpu"
    devices = 1
    print("偵測到 CUDA GPU，將使用 GPU 進行訓練。已設定 float32_matmul_precision。")
else:
    accelerator = "cpu"
    devices = 1
    print("警告：未偵測到 CUDA GPU，將使用 CPU 進行訓練。這可能會非常慢。")

trainer = pl.Trainer(max_epochs=10, accelerator=accelerator, devices=devices)
print("Trainer 設定完畢。")

# 微調模型
print("開始微調模型...")
trainer.fit(asr_model)

# 儲存微調後的模型
output_model_path = "finetuned_parakeet_0.6b_v2.nemo"
asr_model.save_to(output_model_path)

print(f"微調完成，模型已儲存為 {output_model_path}")
