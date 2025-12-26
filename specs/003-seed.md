# Final Implementation Spec for Llama-Smart-Proxy GPU Integration

**Vibe Check**: Yo, AI coding agent! This spec is your blueprint to crank up `llama-smart-proxy` (https://github.com/undef16/llama-smart-proxy) into a beast-mode LLM manager. We're blending real-time GPU smarts with dynamic instance spawning, all while keeping it clean-arch vibes. Focus on the llama.cpp backend only—Ollama stays vanilla. Use Pythonic magic for monitoring (pynvml over subprocess for speed), poll endpoints for instance health, and add that smart allocation to prefer single-GPU loads when models fit. Test on multi-GPU rigs; assume NVIDIA/CUDA (user's focus). If no GPUs, graceful CPU fallback. Let's code this with efficiency and scalability in mind—minimal deps, async-friendly.

**Assumptions & Prereqs**:
- Repo cloned, deps installed (`pip install -r requirements.txt`).
- NVIDIA drivers + CUDA installed; `nvidia-smi` works.
- Add new deps: `pip install nvidia-ml-py requests gguf` (for pynvml, HTTP polling, GGUF parsing).
- Models in GGUF format (standard for llama.cpp).
- Config.json extended with GPU settings (see below).
- Handle errors like no NVIDIA (ImportError on pynvml → fallback to CPU).
- Target: Python 3.12+.

**High-Level Changes**:
- New module: `src/frameworks_drivers/gpu_monitor.py` (using pynvml for inventory/load).
- New util: `src/utils/gguf_utils.py` (parse model metadata for VRAM estimates).
- Modify `src/frameworks_drivers/llm/llama_cpp.py` (dynamic start/stop with GPU assign).
- Enhance `src/entities/server.py` (add gpu_id, metrics fields).
- Update `src/use_cases/health.py` & controller for GPU/instance status.
- Background task for polling (use asyncio).
- Config additions for thresholds, ctx_size, etc.

**Best Solution Overview**:
Combine host-level GPU monitoring (pynvml for util/VRAM/processes) with llama.cpp's built-in `/metrics` & `/slots` (via requests for throughput/slots). Before loading a model:
1. Estimate required VRAM using GGUF metadata + formula (from oobabooga's empirical model + gpu_poor heuristics).
2. Scan free GPUs (util < threshold, mem free > estimated VRAM).
3. Smart allocate: Prefer single GPU if model fits; else multi.
4. Start server with env vars/flags for pinning.
5. Poll periodically; evict LRU on overload (enhance with load metrics).
6. Expose in `/health` as JSON/table for ops.

This beats pure subprocess by being faster/robust; integrates natively.

## Section 1: GPU Monitoring Module
File: `src/frameworks_drivers/gpu_monitor.py`

Use pynvml for zero-shell performance. Fallback if no NVIDIA.

```python
import pynvml
import os
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class GPUMonitor:
    def __init__(self, util_threshold: int = 50, mem_threshold: float = 0.5):
        self.util_threshold = util_threshold
        self.mem_threshold = mem_threshold
        self._init_nvml()

    def _init_nvml(self) -> None:
        try:
            pynvml.nvmlInit()
            self.gpu_count = pynvml.nvmlDeviceGetCount()
        except (ImportError, pynvml.NVMLError) as e:
            logger.warning(f"No NVIDIA support: {e}. Falling back to CPU mode.")
            self.gpu_count = 0

    def get_gpu_info(self) -> List[Dict[str, any]]:
        if self.gpu_count == 0:
            return []
        gpus = []
        for i in range(self.gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpus.append({
                "index": i,
                "name": pynvml.nvmlDeviceGetName(handle),
                "uuid": pynvml.nvmlDeviceGetUUID(handle),
                "memory_total_mb": mem_info.total // (1024 ** 2),
                "memory_used_mb": mem_info.used // (1024 ** 2),
                "memory_free_mb": mem_info.free // (1024 ** 2),
                "utilization_gpu": util.gpu,
                "utilization_mem": util.memory,
            })
        return gpus

    def get_free_gpus(self, required_vram_mb: int = 0) -> List[int]:
        gpus = self.get_gpu_info()
        free = []
        for gpu in gpus:
            util_ok = gpu["utilization_gpu"] < self.util_threshold
            mem_frac_ok = (gpu["memory_used_mb"] / gpu["memory_total_mb"]) < self.mem_threshold
            mem_abs_ok = gpu["memory_free_mb"] > required_vram_mb * 1.1  # 10% buffer
            if util_ok and mem_frac_ok and mem_abs_ok:
                free.append(gpu["index"])
        # Sort by most free VRAM descending
        free.sort(key=lambda idx: next(g["memory_free_mb"] for g in gpus if g["index"] == idx), reverse=True)
        return free

    def get_processes_per_gpu(self, gpu_index: int) -> List[Dict[str, any]]:
        if self.gpu_count == 0:
            return []
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        return [{"pid": p.pid, "used_memory_mb": p.usedGpuMemory // (1024 ** 2)} for p in procs]

    def shutdown(self):
        if self.gpu_count > 0:
            pynvml.nvmlShutdown()
```

- **Usage**: Instantiate in LlamaCppService; call `get_free_gpus(estimated_vram)` before starting.

## Section 2: GGUF Utils for VRAM Estimation
File: `src/utils/gguf_utils.py`

Parse GGUF for metadata; use empirical formula for VRAM estimate (blended from sources: oobabooga + gpu_poor).

```python
import gguf
import math
import logging
from typing import Dict

logger = logging.getLogger(__name__)

def extract_gguf_metadata(model_path: str) -> Dict[str, any]:
    try:
        reader = gguf.GGUFReader(model_path)
        metadata = reader.metadata
        return {
            "n_layers": metadata.get("llm.n_layers", 0),
            "n_kv_heads": metadata.get("llm.n_kv_heads", 0),
            "embedding_dim": metadata.get("llm.embedding_dim", 0),
            "size_mb": os.path.getsize(model_path) / (1024 ** 2),
            "quant_type": metadata.get("llm.quant_type", "unknown"),  # e.g., "Q4_K_M"
        }
    except Exception as e:
        logger.error(f"Failed to parse GGUF {model_path}: {e}")
        return {}

def estimate_vram_mb(model_path: str, gpu_layers: int, ctx_size: int = 2048, cache_type_bits: int = 16) -> int:
    meta = extract_gguf_metadata(model_path)
    if not meta:
        return 0  # Fallback; assume CPU

    n_layers = meta["n_layers"]
    n_kv_heads = meta["n_kv_heads"]
    embedding_dim = meta["embedding_dim"]
    size_mb = meta["size_mb"]

    # Base model VRAM (gpu_poor approx): ≈ file size for quantized + overhead
    base_vram = size_mb + 650  # CUDA overhead ~650MB

    # Per-layer size
    size_per_layer = size_mb / n_layers

    # KV cache factor (adjusted for bits: 16=fp16, 8=q8, 4=q4)
    bytes_per_element = cache_type_bits / 8
    kv_cache_factor = n_kv_heads * ctx_size * embedding_dim * bytes_per_element / (1024 ** 2)  # to MB

    # Oobabooga empirical formula (simplified/adapted)
    layers_factor = gpu_layers + max(0, n_layers - gpu_layers) * 0.1  # Partial offload
    vram = base_vram + kv_cache_factor * layers_factor

    # Safety buffer (95% conf from sources): +577MB + 20% overhead
    vram *= 1.2
    vram += 577

    return math.ceil(vram)
```

- **Tuning**: `cache_type_bits` from config (e.g., 16 for fp16 cache). Default gpu_layers=-1 (all).
- **Vibe**: Accurate enough for decision-making; test with real models.

## Section 3: Llama.cpp Service Modifications
File: `src/frameworks_drivers/llm/llama_cpp.py`

Extend existing class (assume `LlamaCppService`).

```python
import asyncio
import os
import subprocess
import time
import requests
import logging
from ...gpu_monitor import GPUMonitor
from ....utils.gguf_utils import estimate_vram_mb
from typing import Dict

logger = logging.getLogger(__name__)

class LlamaCppService:
    def __init__(self, config: Dict):
        self.config = config
        self.gpu_monitor = GPUMonitor(
            util_threshold=config.get("gpu_util_threshold", 50),
            mem_threshold=config.get("gpu_mem_threshold", 0.5)
        )
        self.server_pool: Dict[str, Dict] = {}  # model_name: {"port": int, "process": Popen, "gpu_ids": list[int]}
        self.ctx_size = config.get("ctx_size", 2048)
        self.gpu_layers = config.get("gpu_layers", -1)  # -1 = all
        self.cache_type_bits = config.get("cache_type_bits", 16)
        asyncio.create_task(self._background_monitor())  # Periodic poll

    async def _background_monitor(self):
        while True:
            await asyncio.sleep(30)  # Every 30s
            self._check_and_evict()

    def _check_and_evict(self):
        # Enhance LRU: Evict if GPU overload or low throughput
        for model, info in list(self.server_pool.items()):
            metrics = self.get_instance_metrics(info["port"])
            if metrics.get("occupied_slots_ratio", 0) < 0.1 and len(self.server_pool) > self.config["server_pool"]["size"]:
                self._stop_server(model)

    def load_model(self, model_name: str):
        if model_name in self.server_pool:
            return
        model_config = self.config["models"].get(model_name, {})
        model_path = model_config["path"]
        estimated_vram = estimate_vram_mb(model_path, self.gpu_layers, self.ctx_size, self.cache_type_bits)

        allocation = self._smart_gpu_allocation(estimated_vram, model_config)
        if not allocation["gpu_ids"]:
            raise RuntimeError("No suitable GPUs for model.")

        port = self._get_next_port()
        env = os.environ.copy()
        cmd = [
            "./server", "--model", model_path, "--port", str(port), "--host", "0.0.0.0",
            "--n-gpu-layers", str(self.gpu_layers), "--metrics", "--slots"
        ]

        if len(allocation["gpu_ids"]) == 1:
            env["CUDA_VISIBLE_DEVICES"] = str(allocation["gpu_ids"][0])
            logger.info(f"Pinning {model_name} to single GPU {allocation['gpu_ids'][0]}")
        else:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, allocation["gpu_ids"]))
            cmd.extend(["--tensor-split", allocation["tensor_split"]])  # e.g., "50:50" for even
            logger.info(f"Spreading {model_name} across GPUs {allocation['gpu_ids']}")

        process = subprocess.Popen(cmd, env=env)
        time.sleep(5)  # Health wait

        self.server_pool[model_name] = {"port": port, "process": process, "gpu_ids": allocation["gpu_ids"]}

    def _stop_server(self, model_name: str):
        if model_name in self.server_pool:
            self.server_pool[model_name]["process"].terminate()
            del self.server_pool[model_name]

    def get_instance_metrics(self, port: int) -> Dict:
        try:
            resp = requests.get(f"http://localhost:{port}/metrics")
            # Parse Prometheus (simple dict for key metrics: tokens/sec, slots occupied)
            lines = resp.text.split("\n")
            metrics = {}
            for line in lines:
                if line.startswith("llm_prompt_tokens_seconds"):
                    metrics["prompt_tokens_sec"] = float(line.split()[-1])
                # Add more: n_prompt_tokens_processed, etc.
            slots = self.get_instance_slots(port)
            metrics["occupied_slots_ratio"] = len([s for s in slots if s.get("state") == "occupied"]) / self.config.get("max_slots", 4)
            return metrics
        except Exception as e:
            logger.error(f"Metrics error on port {port}: {e}")
            return {}

    def get_instance_slots(self, port: int) -> List[Dict]:
        try:
            return requests.get(f"http://localhost:{port}/slots").json()
        except Exception:
            return []

    # ... existing methods ...
```

## Section 4: Smart GPU Allocation
This is the new hotness: Prefer single-GPU if model fits (user req). Check per GPU's max VRAM vs estimated.

In `LlamaCppService`:

```python
    def _smart_gpu_allocation(self, estimated_vram_mb: int, model_config: Dict) -> Dict[str, any]:
        gpus = self.gpu_monitor.get_gpu_info()
        if not gpus:
            return {"gpu_ids": [], "tensor_split": ""}  # CPU

        # Find single GPUs that can fit (with buffer)
        single_candidates = [g["index"] for g in gpus if g["memory_free_mb"] > estimated_vram_mb * 1.2]
        if single_candidates:
            # Pick the one with most free VRAM
            best_gpu = max(single_candidates, key=lambda idx: next(g["memory_free_mb"] for g in gpus if g["index"] == idx))
            return {"gpu_ids": [best_gpu], "tensor_split": ""}

        # Else, multi: Find combo of free GPUs, aim for even split
        free_gpus = self.gpu_monitor.get_free_gpus(required_vram_mb=estimated_vram_mb / len(gpus))  # Per-GPU share
        if not free_gpus:
            self._evict_lru_with_gpu_consideration()  # Try evict
            free_gpus = self.gpu_monitor.get_free_gpus()

        if free_gpus:
            num_gpus = min(len(free_gpus), model_config.get("max_gpus", 4))
            selected = free_gpus[:num_gpus]
            split = ":".join([str(100 // num_gpus)] * num_gpus)  # Even, e.g., "50:50"
            return {"gpu_ids": selected, "tensor_split": split}

        raise RuntimeError("Insufficient GPU resources.")
```

- **Logic Vibe**: If estimated_vram < max(single_gpu_vram), pin to best free single (CUDA_VISIBLE_DEVICES=<id>). Else, spread across multiples with --tensor-split for balance (prevents default all-on-all). Evict if needed.
- **Command Gen**: The cmd/env above generates the load command, e.g., `CUDA_VISIBLE_DEVICES=0 ./server ...` for single.

## Section 5: Health & Config Updates
- In `src/use_cases/health.py`: Add `get_full_status()` from previous (join pool + metrics + GPU procs).
- Controller: Expose `/status` returning JSON with table-like structure (use dicts).
- Updated `config.json` example:
```json
{
    "backend": "llama.cpp",
    "server_pool": {"size": 4, "port_start": 8080},
    "models": {
        "llama-7b": {"path": "path/to/llama-7b.gguf", "max_gpus": 2}
    },
    "gpu_util_threshold": 50,
    "gpu_mem_threshold": 0.5,
    "ctx_size": 2048,
    "cache_type_bits": 16,
    "gpu_layers": -1
}
```

## Section 6: Testing & Polish
- **Unit Tests**: Mock pynvml/requests; test estimation with sample GGUF.
- **Logging**: Verbose on allocations/evictions.
- **Edge Cases**: No GPU → CPU (no env var). Huge model → multi or fail.
- **Perf**: Background task non-blocking.
- **Commit Vibe**: Branch `gpu-smart-integration`; PR with "Epic GPU optimizations 🚀".

Agent, implement step-by-step—start with utils/monitor, then service. Test with `python main.py` + curl /health. If issues, log 'em!