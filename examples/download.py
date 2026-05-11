import os
from huggingface_hub import snapshot_download

models = [
    ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen2.5-0.5B-Instruct"),
    ("yuhuili/EAGLE-LLaMA3.1-Instruct-8B", "EAGLE-LLaMA3.1-Instruct-8B"),
    ("yuhuili/EAGLE-LLaMA3-Instruct-8B", "EAGLE-LLaMA3-Instruct-8B"),
    ("meta-llama/Meta-Llama-3-8B-Instruct", "Meta-Llama-3-8B-Instruct"),
    ("meta-llama/Llama-3.1-8B-Instruct", "Llama-3.1-8B-Instruct"),
    ("RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8", "Meta-Llama-3.1-8B-Instruct-FP8"),
    ("RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8", "Meta-Llama-3.1-8B-Instruct-quantized.w8a8"),
    ("RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a16", "Meta-Llama-3.1-8B-Instruct-quantized.w8a16"),
    ("RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16", "Meta-Llama-3.1-8B-Instruct-quantized.w4a16"),
    ("meta-llama/Meta-Llama-3-70B-Instruct", "Meta-Llama-3-70B-Instruct"),
    ("mistralai/Mixtral-8x7B-Instruct-v0.1", "Mixtral-8x7B-Instruct-v0.1"),
]

hf_token = os.getenv("HF_TOKEN")

os.makedirs("models", exist_ok=True)
for model_path, model_name in models:
    print(model_name)
    for i in range(1000):
        try:
            snapshot_folder = snapshot_download(
                repo_id=model_path, 
                local_dir=f"models/{model_name}", 
                local_dir_use_symlinks=False, 
                resume_download=True, 
                token=hf_token, 
                max_workers=8,
                ignore_patterns=["consolidated*.safetensors", "original/*"]
            )
            break
        except Exception as e:
            print(f"number {i}th failure: {e}")

print("Finished downloading models")