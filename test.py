import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version by PyTorch: {torch.version.cuda}") # 顯示 PyTorch 編譯時使用的 CUDA 版本
    # 比較這個版本和您系統 nvcc --version 的輸出
