# llama.cpp (CUDA) Runner

## Setup
Build `llama-server` with `-DGGML_CUDA=ON` (CUDA 12.8+ for SM120 Blackwell) and install Python deps:
```bash
pip install -r requirements.txt
```

## Usage
```bash
python run.py --runner nvidia_llamacpp_7c50deed --suite suite_F --gguf-path /path/to/Qwen2.5-0.5B-Instruct-BF16.gguf
```
