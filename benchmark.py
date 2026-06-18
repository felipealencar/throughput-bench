#!/usr/bin/env python3
"""Throughput Bench — rigorous throughput benchmarking for geospatial model backbones.

Measures inference throughput (images/sec) for classification across many model
architectures, precision modes, and hardware configurations.

Usage examples:
    # Full benchmark on GPU 0 (bs=512, 30s per config, halves on OOM)
    python benchmark.py --gpu-id 0

    # Quick test with one model
    python benchmark.py --gpu-id 0 --models resnet50

    # Manual batch size sweep
    python benchmark.py --gpu-id 0 --batch-sizes 1 8 32 64

    # AMP precision only
    python benchmark.py --gpu-id 0 --precisions amp

    # Include DataLoader overhead (default is pure GPU compute)
    python benchmark.py --gpu-id 0 --dataloader
"""

import argparse
import csv
import gc
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import timm
import torch
import torch.nn as nn

from data import create_dataloader, make_spectral_batch
from models import ModelConfig, get_models

# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "model_name",
    "display_name",
    "model_family",
    "model_type",
    "precision",
    "compiled",
    "compile_mode",
    "gpu_name",
    "gpu_mem_gb",
    "batch_size",
    "throughput_mean",
    "pixels_per_sec",
    "latency_mean_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "params_M",
    "macs_G",
    "peak_memory_mb",
    "tf32_enabled",
    "input_channels",
    "input_size",
    "pytorch_version",
    "cuda_version",
    "timestamp",
]

# ---------------------------------------------------------------------------
# Hardware helpers
# ---------------------------------------------------------------------------


def get_gpu_name(gpu_id: int = 0) -> str:
    return torch.cuda.get_device_name(gpu_id)


def get_gpu_mem_gb(gpu_id: int = 0) -> float:
    return torch.cuda.get_device_properties(gpu_id).total_memory / 1e9


def get_gpu_slug() -> str:
    """Sanitized GPU name for filenames, e.g. 'tesla_v100_sxm2_32gb'."""
    name = get_gpu_name(0)
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    return slug


def get_cuda_version() -> str:
    return torch.version.cuda or "N/A"


def collect_hardware_info(gpu_id: int) -> dict:
    """Collect full hardware metadata."""
    info = {
        "gpu_name": get_gpu_name(0),
        "gpu_mem_gb": round(get_gpu_mem_gb(0), 1),
        "gpu_id_physical": gpu_id,
        "cuda_version": get_cuda_version(),
        "pytorch_version": torch.__version__,
        "timm_version": timm.__version__,
        "python_version": platform.python_version(),
        "os": platform.system(),
        "cpu": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "cudnn_version": (
            torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        ),
    }

    # Extended GPU metadata via nvidia-smi
    smi_queries = {
        "driver_version": "driver_version",
        "persistence_mode": "persistence_mode",
        "power_limit_w": "power.limit",
        "clock_max_sm_mhz": "clocks.max.sm",
        "clock_max_mem_mhz": "clocks.max.mem",
    }
    for key, query in smi_queries.items():
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--query-gpu={query}",
                    "--format=csv,noheader,nounits",
                    f"--id={gpu_id}",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            info[key] = result.stdout.strip()
        except Exception:
            info[key] = None

    # Git SHA
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        info["git_sha"] = result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        info["git_sha"] = None

    return info


def check_gpu_free(gpu_id: int) -> bool:
    """Check that the target GPU has no other processes running."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        gpu_uuids_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        target_uuid = None
        for line in gpu_uuids_result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[0] == str(gpu_id):
                target_uuid = parts[1]
                break
        if target_uuid is None:
            return True
        our_pid = str(os.getpid())
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2:
                pid, uuid = parts
                if uuid == target_uuid and pid != our_pid:
                    return False
        return True
    except Exception:
        return True


def gpu_cleanup():
    """Free GPU memory between benchmark runs."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch._dynamo.reset()


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def count_params(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def estimate_macs(
    model: nn.Module, input_shape: tuple = (1, 3, 224, 224), device: str = "cpu"
) -> float:
    from torch.utils.flop_counter import FlopCounterMode

    inp = torch.randn(*input_shape, device=device)
    with FlopCounterMode(display=False) as fcm:
        model(inp)
    return fcm.get_total_flops() / 1e9


def _timm_create_kwargs(timm_name: str, input_channels: int, input_size: int) -> dict:
    """Build kwargs for timm.create_model, forcing img_size for ViT-like models.

    ViT-like models embed the image size in their patch embedding and accept
    `img_size`. We always pass it (rather than relying on the model's default)
    so that e.g. `vit_huge_plus_patch16_dinov3` (default 256) gets created at
    our standard input shape instead of whatever the timm config bakes in.
    CNNs are resolution-agnostic and reject the parameter.
    """
    kwargs: dict = {"pretrained": False, "num_classes": 10, "in_chans": input_channels}
    vit_prefixes = (
        "vit_",
        "deit",
        "swin_",
        "beit_",
        "coatnet_",
        "maxvit_",
        "eva_",
    )
    if any(timm_name.startswith(p) for p in vit_prefixes):
        kwargs["img_size"] = input_size
    return kwargs


def create_model(
    config: ModelConfig,
    device: torch.device,
    input_channels: int = 3,
    input_size: int = 224,
) -> nn.Module | None:
    """Instantiate a classification model. Returns None if unsupported."""
    if config.source == "geo":
        from geo_models import create_geo_model

        return create_geo_model(config.geo_model_key, device)

    kwargs = _timm_create_kwargs(config.timm_factory, input_channels, input_size)
    if config.patch_size is not None:
        kwargs["patch_size"] = config.patch_size
    try:
        model = timm.create_model(config.timm_factory, **kwargs)
    except Exception:
        return None  # Model incompatible with this input size
    model = model.to(device)
    model.eval()
    return model


def apply_precision(model: nn.Module, precision: str) -> nn.Module:
    """Cast model parameters/buffers to the target precision in place.

    ``fp32`` and ``amp`` leave the model in fp32 (autocast handles AMP at
    forward time); ``fp16``/``bf16`` cast all params and buffers via
    ``.half()`` / ``.bfloat16()``.
    """
    if precision == "fp16":
        model = model.half()
    elif precision == "bf16":
        model = model.bfloat16()
    return model


def apply_compile(model: nn.Module, compile_mode: str) -> tuple[nn.Module, bool]:
    """Wrap ``model`` with ``torch.compile`` in the requested mode.

    Returns ``(wrapped_or_original_model, ok)``. Pass ``"none"`` to opt out.
    On failure (rare — most ``torch.compile`` errors surface lazily during
    the first forward pass, not here), returns the original model with
    ``ok=False`` so the caller can record ``compile_mode='none'``.
    """
    if compile_mode == "none":
        return model, True
    try:
        model = torch.compile(model, mode=compile_mode)
        return model, True
    except Exception as e:
        print(f"    ⚠ torch.compile({compile_mode}) failed: {e}")
        return model, False


# ---------------------------------------------------------------------------
# Benchmarking core
# ---------------------------------------------------------------------------


def benchmark_gpu(
    model: nn.Module,
    dataloader,
    precision: str,
    device: torch.device,
    num_warmup: int = 20,
    min_timed_seconds: float = 30.0,
) -> dict:
    """Benchmark on GPU using wall-clock timing with DataLoader."""
    use_amp = precision == "amp"

    # --- warmup ---
    data_iter = iter(dataloader)
    for _ in range(num_warmup):
        try:
            images, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            images, _ = next(data_iter)
        images = images.to(device, non_blocking=True)
        if precision == "fp16":
            images = images.half()
        elif precision == "bf16":
            images = images.bfloat16()
        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast("cuda"):
                    _ = model(images)
            else:
                _ = model(images)
    torch.cuda.synchronize()

    # Reset peak memory AFTER warmup so we measure steady-state only
    torch.cuda.reset_peak_memory_stats()

    # --- timed iterations ---
    batch_size = dataloader.batch_size
    total_images = 0
    batch_times: list[float] = []
    data_iter = iter(dataloader)

    torch.cuda.synchronize()
    t_start = time.perf_counter()

    while True:
        try:
            images, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            images, _ = next(data_iter)

        images = images.to(device, non_blocking=True)
        if precision == "fp16":
            images = images.half()
        elif precision == "bf16":
            images = images.bfloat16()

        t_batch = time.perf_counter()
        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast("cuda"):
                    _ = model(images)
            else:
                _ = model(images)
        torch.cuda.synchronize()
        batch_times.append(time.perf_counter() - t_batch)

        total_images += batch_size

        if time.perf_counter() - t_start >= min_timed_seconds:
            break

    elapsed_s = time.perf_counter() - t_start

    return _format_gpu_stats(total_images, batch_size, elapsed_s, batch_times)


def benchmark_gpu_preallocated(
    model: nn.Module,
    batch_size: int,
    precision: str,
    device: torch.device,
    num_warmup: int = 20,
    min_timed_seconds: float = 30.0,
    input_channels: int = 3,
    input_size: int = 224,
    data_mode: str = "randn",
) -> dict:
    """Benchmark on GPU with a pre-allocated batch (no DataLoader overhead)."""

    use_amp = precision == "amp"

    if data_mode == "spectral":
        images = make_spectral_batch(batch_size, input_channels, input_size, device=device)
    elif data_mode == "ones":
        images = torch.ones(batch_size, input_channels, input_size, input_size, device=device)
    else:
        images = torch.randn(batch_size, input_channels, input_size, input_size, device=device)
    if precision == "fp16":
        images = images.half()
    elif precision == "bf16":
        images = images.bfloat16()

    # --- warmup ---
    for _ in range(num_warmup):
        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast("cuda"):
                    _ = model(images)
            else:
                _ = model(images)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()

    # --- timed ---
    total_images = 0
    batch_times: list[float] = []
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    while True:
        t_batch = time.perf_counter()
        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast("cuda"):
                    _ = model(images)
            else:
                _ = model(images)
        torch.cuda.synchronize()
        batch_times.append(time.perf_counter() - t_batch)
        total_images += batch_size
        if time.perf_counter() - t_start >= min_timed_seconds:
            break

    elapsed_s = time.perf_counter() - t_start
    del images

    return _format_gpu_stats(total_images, batch_size, elapsed_s, batch_times)


def _format_gpu_stats(
    total_images: int, batch_size: int, elapsed_s: float, batch_times: list[float]
) -> dict:
    throughput = total_images / elapsed_s
    peak_mem = torch.cuda.max_memory_allocated() / 1e6
    latency_mean_ms = elapsed_s / (total_images / batch_size) * 1000

    if batch_times:
        import numpy as np

        batch_ms = np.array(batch_times) * 1000
        p50 = float(np.percentile(batch_ms, 50))
        p95 = float(np.percentile(batch_ms, 95))
        p99 = float(np.percentile(batch_ms, 99))
    else:
        p50 = p95 = p99 = latency_mean_ms

    return {
        "throughput_mean": float(throughput),
        "latency_mean_ms": float(latency_mean_ms),
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "latency_p99_ms": p99,
        "peak_memory_mb": float(peak_mem),
    }


# ---------------------------------------------------------------------------
# Single benchmark run helper
# ---------------------------------------------------------------------------


def run_single_benchmark(
    mc: ModelConfig,
    precision: str,
    compile_mode: str,
    batch_size: int,
    device: torch.device,
    args,
    gpu_name: str,
    gpu_mem_gb: float,
    macs_g: float,
    params_m: float,
    tf32_enabled: bool = False,
    input_channels: int = 3,
    input_size: int = 224,
) -> dict | None:
    """Run a single benchmark config. Returns a CSV row dict or None."""
    gpu_cleanup()
    model = None
    dl = None
    try:
        model = create_model(
            mc,
            device,
            input_channels=input_channels,
            input_size=input_size,
        )
        if model is None:
            return None
        model = apply_precision(model, precision)
        model, compile_ok = apply_compile(model, compile_mode)
        actual_compile_mode = compile_mode if compile_ok else "none"
        actual_compiled = compile_ok and compile_mode != "none"

        torch.cuda.reset_peak_memory_stats()
        if not args.dataloader:
            stats = benchmark_gpu_preallocated(
                model,
                batch_size,
                precision,
                device,
                num_warmup=args.warmup,
                min_timed_seconds=args.timed_seconds,
                input_channels=input_channels,
                input_size=input_size,
                data_mode=args.data_mode,
            )
        else:
            dl = create_dataloader(
                batch_size=batch_size,
                num_workers=8,
                prefetch_factor=2,
                length=max(batch_size * 500, 10_000),
                channels=input_channels,
                size=input_size,
                data_mode=args.data_mode,
            )
            stats = benchmark_gpu(
                model,
                dl,
                precision,
                device,
                num_warmup=args.warmup,
                min_timed_seconds=args.timed_seconds,
            )

        pixels_per_sec = stats["throughput_mean"] * input_size * input_size
        return {
            "model_name": mc.timm_name,
            "display_name": mc.display_name,
            "model_family": mc.family,
            "model_type": mc.arch_type,
            "precision": precision,
            "compiled": actual_compiled,
            "compile_mode": actual_compile_mode,
            "gpu_name": gpu_name,
            "gpu_mem_gb": f"{gpu_mem_gb:.1f}",
            "batch_size": batch_size,
            "throughput_mean": f"{stats['throughput_mean']:.2f}",
            "pixels_per_sec": f"{pixels_per_sec:.0f}",
            "latency_mean_ms": f"{stats['latency_mean_ms']:.3f}",
            "latency_p50_ms": f"{stats['latency_p50_ms']:.3f}",
            "latency_p95_ms": f"{stats['latency_p95_ms']:.3f}",
            "latency_p99_ms": f"{stats['latency_p99_ms']:.3f}",
            "params_M": f"{params_m:.2f}",
            "macs_G": f"{macs_g:.2f}",
            "peak_memory_mb": f"{stats['peak_memory_mb']:.1f}",
            "tf32_enabled": tf32_enabled,
            "input_channels": input_channels,
            "input_size": input_size,
            "pytorch_version": torch.__version__,
            "cuda_version": get_cuda_version(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    except torch.cuda.OutOfMemoryError:
        return "OOM"
    except RuntimeError as e:
        # torch.compile is lazy — Triton/inductor and dynamo errors surface
        # on first forward pass, not at compile() time.  Catch the whole
        # TorchDynamoException family so that one model failing to compile
        # doesn't kill the entire benchmark sweep.
        from torch._dynamo.exc import TorchDynamoException

        if isinstance(e, TorchDynamoException):
            print(f"\n    ⚠ torch.compile failed at runtime (skipping): {e}")
            return "COMPILE_ERROR"
        msg = str(e)
        if (
            "canUse32BitIndexMath" in msg
            or "32-bit indexing" in msg
            or "INT_MAX" in msg
            or "invalid configuration argument" in msg
        ):
            print("\n    ⚠ CUDA kernel limit exceeded; skipping.")
            return "OOM"
        if "match" in msg and ("size" in msg or "shape" in msg):
            print("\n    ⚠ Incompatible input size; skipping.")
            return None
        raise
    finally:
        # Explicitly shut down DataLoader workers to free file descriptors
        if dl is not None and hasattr(dl, "_iterator") and dl._iterator is not None:
            dl._iterator._shutdown_workers()
        del model, dl
        gpu_cleanup()


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------


def _precision_skip_reason(mc: ModelConfig, precision: str, tf32_enabled: bool) -> str | None:
    """Return a human-readable reason if (mc, precision) cannot run, else None."""
    if precision == "bf16" and not tf32_enabled:
        return "bf16 not supported on this GPU (pre-Ampere)"
    if mc.source == "geo":
        from geo_models import GEO_MODEL_REGISTRY

        entry = GEO_MODEL_REGISTRY.get(mc.geo_model_key, {})
        supported = entry.get("supported_precisions")
        if supported is not None and precision not in supported:
            return f"{precision} not supported by geo model wrapper"
    return None


def _build_plan(
    model_configs: list[ModelConfig],
    args,
    tf32_enabled: bool,
    completed_keys: set[tuple],
    input_channels: int,
    input_size: int,
) -> tuple[list[dict], dict[str, int]]:
    """Enumerate every (model, precision, compile_mode, batch_size) combo and
    split into (run_list, skip_summary).

    `run_list` is a list of task dicts to execute. `skip_summary` is a mapping
    from human-readable reason to count.
    """
    run_list: list[dict] = []
    skip_summary: dict[str, int] = {}

    def bump(reason: str) -> None:
        skip_summary[reason] = skip_summary.get(reason, 0) + 1

    for mc in model_configs:
        if mc.source == "geo":
            ch, sz = mc.native_channels, mc.native_size
        else:
            ch, sz = input_channels, input_size

        for prec in args.precisions:
            prec_skip = _precision_skip_reason(mc, prec, tf32_enabled)
            for cm in args.compile_modes:
                for bs in args.batch_sizes:
                    if prec_skip is not None:
                        bump(prec_skip)
                        continue
                    # NB: resume key intentionally omits batch_size so that
                    # rows produced by the OOM-halving fallback still match.
                    key = (mc.timm_name, prec, cm, ch, sz)
                    if key in completed_keys:
                        bump("already in CSV")
                        continue
                    run_list.append(
                        {
                            "mc": mc,
                            "precision": prec,
                            "compile_mode": cm,
                            "batch_size": bs,
                            "channels": ch,
                            "size": sz,
                        }
                    )

    return run_list, skip_summary


def _print_plan_summary(
    run_list: list[dict],
    skip_summary: dict[str, int],
    total_models: int,
) -> None:
    n_skip = sum(skip_summary.values())
    total = len(run_list) + n_skip
    print()
    print(f"📊 Plan: {total} total configs ({len(run_list)} to run, {n_skip} to skip)")
    if skip_summary:
        for reason, n in sorted(skip_summary.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"   ⏭ {n:4d} × {reason}")
    if run_list:
        models_with_work = {t["mc"].timm_name for t in run_list}
        print(
            f"📦 Models with work: {len(models_with_work)} of {total_models} "
            f"(skipping models with no remaining configs)"
        )
    print()


def _load_completed_keys(output_path: Path) -> set[tuple]:
    """Read existing rows from ``output_path`` and return the resume key set.

    The key is ``(model_name, precision, compile_mode, input_channels,
    input_size)`` — intentionally without ``batch_size`` so rows produced
    by the OOM-halving fallback still match a fresh request at the
    original (larger) batch size.
    """
    if not (output_path.exists() and output_path.stat().st_size > 0):
        return set()
    keys: set[tuple] = set()
    try:
        with open(output_path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    ch = int(row.get("input_channels") or 3)
                    sz = int(row.get("input_size") or 224)
                except ValueError:
                    continue
                keys.add(
                    (
                        row.get("model_name"),
                        row.get("precision"),
                        row.get("compile_mode"),
                        ch,
                        sz,
                    )
                )
    except Exception:
        return set()
    return keys


def _oom_row(
    mc: ModelConfig,
    precision: str,
    compile_mode: str,
    batch_size: int,
    gpu_name: str,
    gpu_mem_gb: float,
) -> dict:
    """Build a CSV row capturing an OOM-at-bs=1 outcome."""
    fixed = {
        "model_name": mc.timm_name,
        "display_name": mc.display_name,
        "model_family": mc.family,
        "model_type": mc.arch_type,
        "precision": precision,
        "compiled": compile_mode != "none",
        "compile_mode": compile_mode,
        "gpu_name": gpu_name,
        "gpu_mem_gb": f"{gpu_mem_gb:.1f}",
        "batch_size": batch_size,
        "throughput_mean": "OOM",
    }
    return {**{c: "" for c in CSV_COLUMNS if c not in fixed}, **fixed}


def run_benchmark(args):
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gpu_id = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")
    gpu_name = get_gpu_name(0)
    gpu_mem_gb = get_gpu_mem_gb(0)
    print(f"🖥  Device: GPU {gpu_id} ({gpu_name}, {gpu_mem_gb:.0f} GB)")

    # Enable TF32 for fp32 matmuls (Ampere+). Matches what virtually all
    # "fp32" benchmarks on modern NVIDIA hardware actually measure; without
    # this, the fp32 precision mode runs strict IEEE-754 and looks
    # artificially slow on H100/A100.
    torch.set_float32_matmul_precision("high")
    tf32_enabled = torch.cuda.get_device_capability(0) >= (8, 0)
    tf32_label = "enabled" if tf32_enabled else "not available"
    print(f'🔢 float32 matmul precision: "high" (TF32 {tf32_label} on this GPU)')

    if not check_gpu_free(gpu_id):
        if args.force:
            print(
                f"⚠  WARNING: Other processes detected on GPU {gpu_id}. "
                f"Results may be unreliable. (--force used, continuing)"
            )
        else:
            print(f"❌ ERROR: Other processes detected on GPU {gpu_id}.")
            print("   Benchmarks require an idle GPU for reliable results.")
            print("   Use --force to override this check.")
            sys.exit(1)

    if args.output == "auto":
        slug = get_gpu_slug()
        output_path = Path(f"results/{slug}.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)

    hw_info = collect_hardware_info(args.gpu_id)
    hw_path = output_path.parent / (output_path.stem + "_hardware.json")
    with open(hw_path, "w") as f:
        json.dump(hw_info, f, indent=2)
    print(f"💾 Hardware info: {hw_path}")

    model_configs = get_models(args.models if args.models else None)
    input_channels = args.input_channels
    input_size = args.input_size
    print(f"📋 Models: {len(model_configs)}")
    print(f"📋 Precisions: {args.precisions}")
    print(f"📋 Compile modes: {args.compile_modes}")
    print(f"📋 Input: {input_channels}×{input_size}×{input_size}")
    print(f"📋 Batch sizes: {args.batch_sizes} (halve on OOM until it fits)")
    print(f"📋 Timed seconds: {args.timed_seconds}")
    print(f"📋 Data mode: {args.data_mode}")

    # Load already-completed configs and decide up front what to run vs skip,
    # so a resumed benchmark doesn't pay any per-config / per-model overhead
    # for work that's already been done.
    completed_keys = _load_completed_keys(output_path)
    if completed_keys:
        print(f"📂 Found {len(completed_keys)} existing configs in {output_path}")

    run_list, skip_summary = _build_plan(
        model_configs, args, tf32_enabled, completed_keys, input_channels, input_size
    )
    _print_plan_summary(run_list, skip_summary, total_models=len(model_configs))

    if not run_list:
        print("✅ Nothing to run.")
        return

    # Open CSV writer
    file_exists = output_path.exists() and output_path.stat().st_size > 0
    csv_file = open(output_path, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
    if not file_exists:
        writer.writeheader()

    # Group remaining tasks by model so we compute MACs once per model that
    # has at least one config to run (no waste on fully-skipped models).
    plan_by_model: dict[str, list[dict]] = {}
    for task in run_list:
        plan_by_model.setdefault(task["mc"].timm_name, []).append(task)

    completed = 0
    total = len(run_list)

    for model_name, tasks in plan_by_model.items():
        mc = tasks[0]["mc"]
        model_channels = tasks[0]["channels"]
        model_size = tasks[0]["size"]

        print(f"\n{'=' * 70}")
        print(f"  {mc.display_name} ({mc.timm_name}) — {mc.arch_type}")
        print(f"{'=' * 70}")

        # Compute MACs once per model (CPU). If the model can't even be
        # instantiated for this input shape, skip all of its tasks.
        macs_g, params_m = -1.0, -1.0
        try:
            if mc.source == "geo":
                from geo_models import create_geo_model

                tmp = create_geo_model(mc.geo_model_key, torch.device("cpu"))
            else:
                tmp_kwargs = _timm_create_kwargs(mc.timm_factory, model_channels, model_size)
                if mc.patch_size is not None:
                    tmp_kwargs["patch_size"] = mc.patch_size
                tmp = timm.create_model(mc.timm_factory, **tmp_kwargs)
        except Exception as e:
            print(
                f"  ⏭ Skipping all {len(tasks)} configs (model incompatible "
                f"with {model_channels}ch × {model_size}×{model_size}): {e}"
            )
            completed += len(tasks)
            continue

        try:
            tmp.eval()
            params_m = count_params(tmp)
            cls_shape = (1, model_channels, model_size, model_size)
            macs_g = estimate_macs(tmp, input_shape=cls_shape, device="cpu")
        except Exception as e:
            print(f"  ⚠ Could not compute MACs: {e}")
        finally:
            del tmp
            gc.collect()

        for task in tasks:
            completed += 1
            prec = task["precision"]
            cm = task["compile_mode"]
            requested_bs = task["batch_size"]

            # OOM-halving loop: start at requested bs, halve on OOM until
            # the batch fits or we hit bs=1.
            bs = requested_bs
            label = f"  [{completed}/{total}] {prec} | compile={cm} | bs={bs}"
            print(label, end=" ... ", flush=True)
            result = run_single_benchmark(
                mc,
                prec,
                cm,
                bs,
                device,
                args,
                gpu_name,
                gpu_mem_gb,
                macs_g,
                params_m,
                tf32_enabled=tf32_enabled,
                input_channels=model_channels,
                input_size=model_size,
            )
            while result == "OOM" and bs > 1:
                bs //= 2
                print(f"OOM → retrying bs={bs}", end=" ... ", flush=True)
                result = run_single_benchmark(
                    mc,
                    prec,
                    cm,
                    bs,
                    device,
                    args,
                    gpu_name,
                    gpu_mem_gb,
                    macs_g,
                    params_m,
                    tf32_enabled=tf32_enabled,
                    input_channels=model_channels,
                    input_size=model_size,
                )

            if result == "COMPILE_ERROR":
                print("SKIP (compile error)")
                continue

            if result == "OOM":
                print("OOM (even at bs=1)")
                writer.writerow(_oom_row(mc, prec, cm, bs, gpu_name, gpu_mem_gb))
                csv_file.flush()
                continue

            if result is None:
                print("SKIP")
                continue

            tp = float(result["throughput_mean"])
            print(f"{tp:.1f} img/s")
            writer.writerow(result)
            csv_file.flush()

            time.sleep(1)

        time.sleep(2)

    csv_file.close()
    print(f"\n✅ Results saved to {output_path}")
    print(f"   {completed} configurations processed ({total} planned)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Throughput Bench: Geospatial model throughput benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--gpu-id", type=int, default=0, help="GPU index to use (default: 0)")
    p.add_argument("--models", nargs="+", default=None, help="Filter to specific timm model names")
    p.add_argument(
        "--precisions",
        nargs="+",
        default=["fp32", "fp16", "amp", "bf16"],
        choices=["fp32", "fp16", "amp", "bf16"],
        help="Precision modes (bf16 auto-skipped on pre-Ampere GPUs)",
    )
    p.add_argument(
        "--compile-modes",
        nargs="+",
        default=["none", "default"],
        choices=["none", "default", "max-autotune"],
        help="torch.compile modes to benchmark (default: none + default)",
    )
    p.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[512],
        help="Batch sizes to run (default: 512). Each value halves on OOM until it fits.",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Number of warmup iterations (default: 20)",
    )
    p.add_argument(
        "--timed-seconds",
        type=float,
        default=30.0,
        help="Minimum seconds to time (default: 30)",
    )
    p.add_argument(
        "--output",
        type=str,
        default="auto",
        help="Output CSV path (default: auto-detect from GPU)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Run even if other processes are using the GPU",
    )
    p.add_argument(
        "--dataloader",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a PyTorch DataLoader to feed data (adds realistic pipeline "
        "overhead). Default is a pre-allocated GPU batch, which measures "
        "peak compute throughput.",
    )
    p.add_argument(
        "--input-channels",
        type=int,
        default=3,
        help="Number of input channels (default: 3). Use 4/6/13 for multispectral EO data.",
    )
    p.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="Spatial input size (default: 224). Images are input_size × input_size.",
    )
    p.add_argument(
        "--data-mode",
        default="randn",
        choices=["ones", "randn", "spectral"],
        help=(
            "Synthetic input distribution (default: randn). "
            "'ones': constant tensor — fast but degenerate for ViT attention. "
            "'randn': standard-normal noise. "
            "'spectral': per-band normal samples approximating S2/Landsat surface reflectance "
            "— recommended for ViT-based geospatial FMs."
        ),
    )
    p.add_argument(
        "--geo-compare",
        action="store_true",
        help="Run geo foundation model comparison: benchmarks all geo models at "
        "native input settings, then re-benchmarks timm models at each unique "
        "geo input config (channels × size) for fair comparison. "
        "Forces fp32 for both halves of the comparison; honors --compile-modes. "
        "To benchmark a geo model at fp16/amp/bf16, call benchmark.py directly "
        "(per-model precision support comes from "
        "geo_models.GEO_MODEL_REGISTRY[...]['supported_precisions']).",
    )
    return p.parse_args()


def run_geo_compare(args):
    """Run geo foundation model comparison benchmark.

    For each unique (channels, size) among geo models, benchmarks:
    1. All geo models with that input config
    2. All timm models at that same (channels, size) for fair comparison
    """
    from models import get_models

    all_models = get_models()
    geo_models = [m for m in all_models if m.source == "geo"]
    timm_models = [m for m in all_models if m.source == "timm"]

    input_configs: dict[tuple[int, int], list[str]] = {}
    for m in geo_models:
        key = (m.native_channels, m.native_size)
        input_configs.setdefault(key, []).append(m.timm_name)

    # Force fp32 for both halves so the geo/timm comparison is apples-to-apples;
    # to benchmark a geo model at fp16/amp/bf16, call benchmark.py directly with
    # --models <name> --precisions ... (per-model supported_precisions filters).
    args.precisions = ["fp32"]

    print("=" * 70)
    print("  Phase 1: Geo foundation models (native input settings)")
    print("=" * 70)
    args.models = [m.timm_name for m in geo_models]
    run_benchmark(args)

    timm_names = [m.timm_name for m in timm_models]
    for (ch, sz), geo_names in sorted(input_configs.items()):
        print()
        print("=" * 70)
        print(f"  Phase 2: Timm models at {ch}ch × {sz}×{sz}")
        print(f"  (matching: {', '.join(geo_names)})")
        print("=" * 70)
        args.models = timm_names
        args.input_channels = ch
        args.input_size = sz
        run_benchmark(args)


if __name__ == "__main__":
    args = parse_args()
    if args.geo_compare:
        run_geo_compare(args)
    else:
        run_benchmark(args)
