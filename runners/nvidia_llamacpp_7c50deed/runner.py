"""
AccelMark — NVIDIA llama.cpp (llama-server CUDA) benchmark runner.

Launches a local `llama-server` subprocess with continuous batching enabled
(`--cont-batching`, `--parallel N`) and drives both offline batch throughput
and online streaming TTFT/TPOT measurements via its native HTTP/SSE `/completion`
endpoint (preserving AccelMark's exact chat-template formatting).

Model resolution for GGUF:
  1. Explicit `--gguf-path /path/to/model.gguf` or `runner_config.gguf_path`
  2. If `model_path` itself is a `.gguf` file or directory containing a `.gguf` file
  3. Optional fallback via `LLAMA_CPP_GGUF_MAP` JSON/YAML or `runner_config.gguf_map`
"""

import asyncio
import atexit
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import httpx
from transformers import AutoTokenizer

from runners.benchmark_runner import BenchmarkRunner, InferenceRequest
from loadgen.types import InferenceResult


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class LlamaCppServerRunner(BenchmarkRunner):
    """AccelMark benchmark runner using llama.cpp (llama-server CUDA)."""

    SUPPORTS_STREAMING = True
    SUPPORTS_BATCHING = True
    SUPPORTS_ONLINE = True
    SUPPORTS_MULTI_CHIP = True

    SUPPORTED_PRECISIONS = ["bf16", "fp16", "fp32"]
    SUPPORTED_QUANTIZATION_BACKENDS = []

    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None
        self._stderr_file = None
        self._stderr_path: Optional[Path] = None
        self._port: int = 0
        self._base_url: str = ""
        self._max_tokens: int = 512
        self._parallel_slots: int = 32
        self._server_bin: str = "llama-server"
        self._gguf_path_override: Optional[str] = None
        self._effective_dtype: Optional[str] = None
        self.tokenizer = None
        atexit.register(self.release_resources)

    def _get_chip_count(self) -> int:
        try:
            import torch
            n = torch.cuda.device_count()
            return n if n > 0 else 1
        except Exception:
            return 1

    def _get_framework_name(self) -> str:
        return "llama.cpp"

    def _get_framework_version(self) -> str:
        binary = shutil.which(self._server_bin) or self._server_bin
        try:
            out = subprocess.check_output(
                [binary, "--version"], stderr=subprocess.STDOUT, text=True, timeout=5
            )
            for line in out.splitlines():
                line = line.strip()
                if "version:" in line.lower():
                    return line.split(":", 1)[1].strip().split()[0]
            return out.strip().splitlines()[0][:40] if out.strip() else "unknown"
        except Exception:
            return "unknown"

    def _resolve_gguf_path(self, model_path: str) -> str:
        cfg = getattr(self, "_runner_config", {})
        if self._gguf_path_override:
            return self._gguf_path_override
        if cfg.get("gguf_path"):
            return str(cfg["gguf_path"])
        gguf_map = cfg.get("gguf_map") or {}
        if model_path in gguf_map:
            return str(gguf_map[model_path])

        p = Path(model_path)
        if p.is_file() and p.suffix.lower() == ".gguf":
            return str(p)
        if p.is_dir():
            ggufs = sorted(p.glob("*.gguf"))
            if ggufs:
                return str(ggufs[0])

        raise FileNotFoundError(
            f"llama.cpp requires a GGUF checkpoint. Provide --gguf-path /path/to/model.gguf "
            f"(or set gguf_path / gguf_map in runner config YAML). Received model_path={model_path!r}"
        )

    def load_model(self, model_path: str, parallelism: dict) -> None:
        cfg = getattr(self, "_runner_config", {})
        self._server_bin = cfg.get("server_bin") or os.environ.get("LLAMA_SERVER_BIN", "llama-server")
        binary = shutil.which(self._server_bin) or self._server_bin
        if not Path(binary).exists() and shutil.which(binary) is None:
            raise FileNotFoundError(
                f"Could not find `{self._server_bin}` in PATH. Build llama.cpp with -DGGML_CUDA=ON "
                f"or set LLAMA_SERVER_BIN=/path/to/llama-server."
            )

        gguf_path = self._resolve_gguf_path(model_path)
        self._max_tokens = int(parallelism["max_tokens"])
        max_model_len = int(parallelism.get("max_model_len") or cfg.get("ctx_per_slot", 4096))
        tp_size = int(parallelism.get("tensor_parallel_size", 1))
        self._parallel_slots = int(cfg.get("parallel_slots", 64 if max_model_len <= 2048 else 16))
        total_ctx = int(cfg.get("ctx_size") or (max_model_len * self._parallel_slots))

        effective_precision = getattr(self, "_effective_precision", "BF16").upper()
        self._effective_dtype = effective_precision.lower()

        # Load HF tokenizer when available so chat template formatting matches vLLM/SGLang
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        except Exception:
            self.tokenizer = None

        self._port = _find_free_port()
        self._base_url = f"http://127.0.0.1:{self._port}"

        threads = int(cfg.get("threads", min(8, os.cpu_count() or 8)))
        cmd = [
            binary,
            "-m", gguf_path,
            "--host", "127.0.0.1",
            "--port", str(self._port),
            "-ngl", str(cfg.get("n_gpu_layers", 999)),
            "-np", str(self._parallel_slots),
            "-c", str(total_ctx),
            "-t", str(threads),
            "-tb", str(threads),
            "--cont-batching",
            "--flash-attn", str(cfg.get("flash_attn", "on")),
        ]
        if tp_size > 1:
            cmd += ["--split-mode", str(cfg.get("split_mode", "row"))]
        for extra_arg in cfg.get("extra_server_args") or []:
            cmd.append(str(extra_arg))

        # Write server stderr to a temp file (NEVER subprocess.PIPE — pipe buffer fills and deadlocks!)
        fd, log_path = tempfile.mkstemp(prefix="accelmark_llamacpp_", suffix=".log")
        self._stderr_path = Path(log_path)
        self._stderr_file = os.fdopen(fd, "w", encoding="utf-8")

        print(f"Launching llama-server on {self._base_url}: {' '.join(cmd)}")
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=self._stderr_file,
        )

        # Wait until /health returns 200
        timeout_s = int(cfg.get("startup_timeout_s", 300))
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            if self._proc.poll() is not None:
                self._stderr_file.flush()
                err_tail = self._stderr_path.read_text(encoding="utf-8", errors="replace")[-2000:]
                raise RuntimeError(
                    f"llama-server exited prematurely (rc={self._proc.returncode}):\n{err_tail}"
                )
            try:
                resp = httpx.get(f"{self._base_url}/health", timeout=2.0)
                if resp.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(0.5)

        self.release_resources()
        raise TimeoutError(f"llama-server did not become healthy within {timeout_s}s")

    def get_effective_dtype(self) -> Optional[str]:
        return self._effective_dtype

    async def _run_single_completion(
        self, client: httpx.AsyncClient, prompt: str, max_tokens: int
    ) -> tuple[str, int, int, bool]:
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": 0.0,
            "top_k": 1,
            "stream": False,
            "cache_prompt": False,
        }
        resp = await client.post(f"{self._base_url}/completion", json=payload)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("content", "") or ""
        out_toks = int(data.get("tokens_predicted") or data.get("timings", {}).get("predicted_n") or 0)
        in_toks = int(data.get("tokens_evaluated") or data.get("timings", {}).get("prompt_n") or 0)
        return text, out_toks, in_toks, True

    def inference_fn_offline(self, requests: list[InferenceRequest]) -> list[InferenceResult]:
        formatted = [self._format_prompt(r.prompt) for r in requests]

        async def _run_batch():
            limits = httpx.Limits(
                max_connections=max(len(formatted), 64),
                max_keepalive_connections=max(len(formatted), 64),
            )
            timeout = httpx.Timeout(timeout=1800.0, connect=30.0)
            async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
                tasks = [
                    self._run_single_completion(
                        client,
                        prompt,
                        getattr(req, "max_tokens", None) or self._max_tokens,
                    )
                    for req, prompt in zip(requests, formatted)
                ]
                return await asyncio.gather(*tasks, return_exceptions=True)

        t_start = time.perf_counter()
        raw_outputs = asyncio.run(_run_batch())
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        results: list[InferenceResult] = []
        acc_texts: list[str] = []
        for item in raw_outputs:
            if isinstance(item, Exception):
                acc_texts.append("")
                results.append(
                    InferenceResult(
                        first_token_time_ms=None,
                        total_time_ms=elapsed_ms,
                        output_tokens=0,
                        input_tokens=0,
                        success=False,
                        output_text="",
                    )
                )
            else:
                text, out_toks, in_toks, ok = item
                acc_texts.append(text)
                results.append(
                    InferenceResult(
                        first_token_time_ms=None,
                        total_time_ms=elapsed_ms,
                        output_tokens=out_toks,
                        input_tokens=in_toks,
                        success=ok,
                        output_text=text,
                    )
                )

        self._last_accuracy_outputs = acc_texts
        return results

    async def inference_fn_streaming(self, request: InferenceRequest) -> InferenceResult:
        formatted = self._format_prompt(request.prompt)
        max_tokens = getattr(request, "max_tokens", None) or self._max_tokens
        payload = {
            "prompt": formatted,
            "n_predict": max_tokens,
            "temperature": 0.0,
            "top_k": 1,
            "stream": True,
            "cache_prompt": False,
        }
        t_start = time.perf_counter()
        first_token_time_ms: Optional[float] = None
        chunks: list[str] = []
        out_toks = 0
        in_toks = 0

        timeout = httpx.Timeout(timeout=600.0, connect=30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", f"{self._base_url}/completion", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if not body or body == "[DONE]":
                        continue
                    evt = json.loads(body)
                    delta = evt.get("content", "") or ""
                    if delta:
                        if first_token_time_ms is None:
                            first_token_time_ms = (time.perf_counter() - t_start) * 1000
                        chunks.append(delta)
                    if evt.get("stop"):
                        out_toks = int(
                            evt.get("tokens_predicted")
                            or evt.get("timings", {}).get("predicted_n")
                            or len(chunks)
                        )
                        in_toks = int(
                            evt.get("tokens_evaluated")
                            or evt.get("timings", {}).get("prompt_n")
                            or 0
                        )

        output_text = "".join(chunks)
        if out_toks == 0 and output_text:
            out_toks = len(chunks)
        return InferenceResult(
            first_token_time_ms=first_token_time_ms,
            total_time_ms=(time.perf_counter() - t_start) * 1000,
            output_tokens=out_toks,
            input_tokens=in_toks,
            success=True,
            output_text=output_text,
        )

    async def inference_fn_token_stream(self, request: InferenceRequest):
        formatted = self._format_prompt(request.prompt)
        payload = {
            "prompt": formatted,
            "n_predict": getattr(request, "max_tokens", None) or self._max_tokens,
            "temperature": 0.0,
            "top_k": 1,
            "stream": True,
            "cache_prompt": False,
        }
        timeout = httpx.Timeout(timeout=600.0, connect=30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", f"{self._base_url}/completion", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if not body or body == "[DONE]":
                        continue
                    evt = json.loads(body)
                    delta = evt.get("content", "") or ""
                    if delta:
                        yield delta

    def get_peak_memory_gb(self) -> Optional[float]:
        # llama-server runs in a separate process; query nvidia-smi used memory as best-effort
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True,
                timeout=3,
            )
            vals = [float(x.strip()) for x in out.splitlines() if x.strip()]
            return round(max(vals) / 1024.0, 3) if vals else None
        except Exception:
            return None

    def release_resources(self) -> None:
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=15)
            except Exception:
                try:
                    self._proc.kill()
                    self._proc.wait(timeout=5)
                except Exception:
                    pass
            self._proc = None
        if self._stderr_file is not None:
            try:
                self._stderr_file.close()
            except Exception:
                pass
            self._stderr_file = None
        if self._stderr_path is not None:
            try:
                self._stderr_path.unlink(missing_ok=True)
            except Exception:
                pass
            self._stderr_path = None

    def parse_args(self):
        args = super().parse_args()
        import argparse

        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--tensor-parallel-size", type=int, default=None, dest="tensor_parallel_size")
        parser.add_argument("--gguf-path", type=str, default=None, dest="gguf_path")
        extra, _ = parser.parse_known_args()

        tp_size, _tp_source = self._resolve_tensor_parallel_size(extra.tensor_parallel_size)
        self._gguf_path_override = extra.gguf_path
        print(f"  tensor_parallel_size = {tp_size}  [{_tp_source}]")

        self._parallelism = {
            "tensor_parallel_size": tp_size,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": 1,
            "data_parallel_size": 1,
        }
        self._chip_count = tp_size
        self._precision = getattr(args, "precision", None)
        return args

    def get_extra_subprocess_args(self, args) -> list[str]:
        extra = [
            "--tensor-parallel-size",
            str(self._parallelism.get("tensor_parallel_size", 1)),
        ]
        if self._gguf_path_override:
            extra += ["--gguf-path", self._gguf_path_override]
        return extra


if __name__ == "__main__":
    LlamaCppServerRunner().main()
