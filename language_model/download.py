import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # ✅ 正确写法

from huggingface_hub import snapshot_download

model_name = "Qwen/Qwen3-Embedding-0.6B"
snapshot_download(repo_id=model_name, local_dir="./Qwen3-Embedding-0.6B")