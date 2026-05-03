# Throughput Bench

**Jump to: [Quick start](#quick-start) | [Models](#models) | [Methodology](#methodology) | [Pixels → km²](#pixelssec--square-kilometers) | [Contributing results](#contributing-results) | [Citation](#citation)**

A small CLI for measuring inference throughput (img/s) of vision backbones on a single GPU, with a focus on geospatial workloads. Covers **29 timm architectures** (ResNet, EfficientNet, ConvNeXt, MobileNet, RegNet, ViT, DeiT3, Swin, BEiT, CoAtNet) plus **12 geospatial foundation-model encoders** (DOFA, CROMA, SenPaMAE, Galileo, OlmoEarth) under fp32 / fp16 / bf16 / AMP and `torch.compile`. Results are appended to a per-GPU CSV ([NVIDIA H100 NVL](results/nvidia_h100_nvl.csv) · [Tesla V100-SXM2 32 GB](results/tesla_v100_sxm2_32gb.csv) so far); an interactive [Globe Race webapp](webapp/index.html) turns those numbers into "how fast can each backbone map the world?"

<p align="center">
  <img src="images/throughput_bench.png" alt="Throughput Bench Globe Race screenshot" width="820">
</p>

**Figure 1.** [Globe Race webapp](webapp/index.html) — pick two backbones and a GSD; the dot grid fills in proportional to land area each model has processed at its measured throughput.

## Quick start

```bash
git clone https://github.com/calebrob6/throughput-bench.git
cd throughput-bench
make setup          # conda env, or: pip install -r requirements.txt
make benchmark      # appends to results/<gpu_slug>.csv (auto-detected from nvidia-smi)
```

Pass extra flags through `ARGS=`:

```bash
make benchmark GPU_ID=2
make benchmark ARGS="--models resnet18 olmoearth_nano --timed-seconds 10"
make benchmark ARGS="--compile-modes default max-autotune"
make benchmark ARGS="--input-channels 4 --input-size 128"
make benchmark ARGS="--geo-compare"     # geo FMs + timm baselines at matching input shapes
```

Re-running on the same GPU is a free no-op for already-completed configs: the script enumerates every `(model, precision, compile, channels, size)` combo up front, prints a one-line skip summary, and runs only what's missing.

### Geospatial foundation models (optional)

The geo wrappers in `geo_models.py` need extra dependencies (Python ≥ 3.11, < 3.14). `make setup` pulls them in via `environment.yml`; for manual install:

```bash
pip install omegaconf
pip install git+https://github.com/allenai/olmoearth_pretrain_minimal.git@main
pip install git+https://github.com/geobreeze/geobreeze.git
```

The git pin on `olmoearth_pretrain_minimal` is intentional — the PyPI release (≤ 0.0.3) misses the dtype-safe `CompositeEncodings` fix from [PR #10](https://github.com/allenai/olmoearth_pretrain_minimal/pull/10), without which OlmoEarth crashes under `.half()` / `.bfloat16()`. Per-model precision support lives in `geo_models.GEO_MODEL_REGISTRY[name]['supported_precisions']` (DOFA / CROMA / Galileo are fp32 + AMP only because of upstream dtype issues; OlmoEarth + SenPaMAE support all four).

### `torch.compile` system requirements

Triton shells out to the system `gcc`, which needs the GNU assembler (`as`) and glibc dev headers. Slim Linux images often miss them and fail with `cannot execute 'as'` or `cannot find /usr/lib/libc_nonshared.a`. Install:

```bash
sudo apt-get install -y binutils libc6-dev      # Debian / Ubuntu
sudo tdnf install -y binutils glibc-devel       # Azure Linux / RHEL family
```

`environment.yml` also pulls `binutils` into the conda env as a fallback.

## Models

| Family | Sizes | Type |
|---|---|---|
| ResNet | 18 / 50 / 101 / 152 | CNN |
| EfficientNet | B0 / B4 / B7 | CNN |
| ConvNeXt | T / S / B / L | CNN |
| MobileNetV3 | Small / Large | CNN |
| RegNetY | 400MF / 4GF | CNN |
| ViT | Ti/16 / S/16 / B/16 / L/16 | ViT |
| DeiT3 | S/16 / B/16 | ViT |
| Swin | T / S / B / L | ViT |
| BEiT | B/16 / L/16 | ViT |
| CoAtNet | 0 / 2 | Hybrid |
| DOFA | B/16 / L/16 | Geo ViT |
| CROMA | Optical / SAR | Geo ViT |
| SenPaMAE | B/16 | Geo ViT |
| Galileo | Nano/8 / Base/8 / Large/8 | Geo ViT |
| OlmoEarth | Nano/8 / Tiny/8 / Base/8 / Large/8 | Geo ViT |

Add new architectures by extending `MODEL_REGISTRY` in `models.py` (timm-compatible names) or `GEO_MODEL_REGISTRY` in `geo_models.py` (custom wrappers).

## Methodology

- **GPU isolation** — aborts if other processes are using the target GPU; override with `--force`.
- **Precision** — `fp32` enables TF32 matmuls on Ampere+ (CSV's `tf32_enabled` flag disambiguates); `fp16` / `bf16` cast the model with `.half()` / `.bfloat16()`; `amp` keeps the model in fp32 and wraps forward in `torch.autocast`. `bf16` is auto-skipped on pre-Ampere GPUs.
- **Compile** — runs both `none` and `default` `torch.compile` modes by default; `max-autotune` available via `--compile-modes`.
- **Batch size** — starts at the requested size (default 512) and halves on OOM until it fits or hits 1. Pass `--batch-sizes 1 8 32 64` to sweep.
- **Timing** — 20 warmup iters, then ≥ 30 s of timed iters, wall-clock with `torch.cuda.synchronize()` at boundaries. Reports throughput, mean / p50 / p95 / p99 latency, peak GPU memory.
- **Data** — pre-allocated GPU batch by default (peak compute throughput); pass `--dataloader` for the realistic end-to-end path that includes DataLoader IPC + host→device transfer. On a V100 the IPC overhead alone (~140 ms / batch at bs=512) roughly halves ResNet-18 throughput vs the pre-allocated path.
- **Reproducibility** — every run also writes `results/<gpu_slug>_hardware.json` with driver / clocks / power-cap / git SHA.

`benchmark_sanity_check.py` is a minimal standalone ResNet-18 timer for cross-checking against the main script.

## Pixels/sec → Square Kilometers

Throughput becomes a coverage rate once you fix the **Ground Sample Distance** (the physical size of one pixel):

```
area_per_patch = (224 × GSD)² / 10⁶  km²
coverage_rate  = throughput × area_per_patch  km²/s
```

| Sensor | GSD | Area / 224² patch | @ 1,000 img/s | @ 5,000 img/s |
|---|---|---|---|---|
| High-res commercial | 0.3 m | 0.0045 km² | 4.5 km²/s | 22.6 km²/s |
| NAIP / aerial | 1 m | 0.050 km² | 50 km²/s | 251 km²/s |
| Sentinel-2 (10m) | 10 m | 5.02 km² | 5,017 km²/s | 25,088 km²/s |
| Sentinel-2 (20m) | 20 m | 20.07 km² | 20,070 km²/s | 100,352 km²/s |
| Landsat (30m) | 30 m | 45.16 km² | 45,158 km²/s | 225,792 km²/s |

Numbers assume non-overlapping patches on a single GPU; sliding windows in production typically overlap 50%+.

## Contributing results

Got an A100, H100, MI300, or anything else? PRs welcome.

```bash
make benchmark            # writes results/<gpu_slug>.csv + _hardware.json
# Add an entry for the new GPU to results/index.json (csv / hardware / label)
git checkout -b results/<your-gpu>
git add results/
git commit -m "Add results for <your GPU>"
git push origin results/<your-gpu>
```

PR checklist:
- [ ] GPU was idle during the run (the script enforces this unless you pass `--force`).
- [ ] `results/index.json` updated so the webapp picks the new GPU up.

## Citation

```bibtex
@software{throughput-bench2026,
  title={Throughput Bench: Geospatial Model Throughput Benchmark},
  author={Robinson, Caleb},
  year={2026},
  url={https://github.com/calebrob6/throughput-bench},
  license={MIT}
}
```

## License

[MIT](LICENSE)
