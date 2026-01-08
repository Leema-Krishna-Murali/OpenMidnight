# OpenMidnight Training Performance Changes

This document summarizes the code/config changes I made to improve training throughput, plus the reasoning and tradeoffs behind each decision. It focuses on FSDP behavior, mixed-precision handling, `torch.compile`, and data pipeline throughput.

## Goals and Constraints
- Increase tokens/sec without breaking checkpoint compatibility or model outputs.
- Keep changes config-driven (prefer toggles over invasive code changes).
- Avoid state_dict key changes when using `torch.compile`.
- Support LoRA/PEFT (but keep changes general to both DINOv2 and LeJEPAv2).
- Preserve existing FSDP wrapping patterns and DINOv2 training flow.

## High-Level Decisions
- **Make compile optional and safe**: `torch.compile` can change state_dict keys if you compile modules directly. To avoid checkpoint breakage, compile callables (wrappers) and keep the module objects unchanged.
- **Prefer bf16 without GradScaler**: GradScaler is only useful for fp16. For bf16, it adds overhead and can complicate optimizer steps without gains.
- **Keep FSDP configurable**: FSDP’s performance depends on a few knobs (`forward_prefetch`, `limit_all_gathers`, etc.). Expose these in config so tuning is easy.
- **Fix cudagraph pitfalls**: With multiple compiled calls per step (teacher + student), cudagraphs can recycle outputs too aggressively. We added step markers and later defaulted to disabling cudagraphs for stability.
- **Expose data loader and streaming knobs**: IO bottlenecks were visible; raising worker count/prefetch was requested, so these are now configurable.

## Code Changes (Specifics + Rationale)

### 1) `torch.compile` integration without breaking checkpoints
Files:
- `dinov2/train/ssl_meta_arch.py`
- `dinov2/train/train.py`

What changed:
- Added `_compiled_callables` to `SSLMetaArch` and `LeJEPAMetaArch`, and `_call_module()` to route calls through compiled wrappers when enabled.
- Added `maybe_compile()` to opt into compilation after FSDP wrapping.
- `train.py` calls `model.maybe_compile()` right after `prepare_for_distributed_training()`, so the compiled graphs reflect the FSDP-wrapped modules.

Why:
- Compiling the module object changes the state_dict key names (e.g., `_orig_mod.*`), which breaks checkpoint and resume logic. Using wrapper functions keeps module identity stable.
- Compiling after FSDP wrapping avoids compiling a different module graph than what actually runs.

### 2) GradScaler gating based on precision
Files:
- `dinov2/train/ssl_meta_arch.py`

What changed:
- Added `_should_use_grad_scaler()` to disable GradScaler when `param_dtype` is not fp16 (e.g., bf16).

Why:
- GradScaler is only needed for fp16 underflow. For bf16 it adds overhead and can stall performance; turning it off is a low-risk speed win.

### 3) CUDAGraph safety and compile options
Files:
- `dinov2/train/train.py`
- `dinov2/train/ssl_meta_arch.py`
- `dinov2/configs/ssl_default_config.yaml`
- `dinov2/configs/train/vitg14_reg4.yaml`

What changed:
- Added `torch.compiler.cudagraph_mark_step_begin()` at the start of each iteration when compile is enabled.
- Added config-driven compile options (`train.compile.*`) including `use_cudagraphs` and `cudagraph_trees`.
- Added `_apply_inductor_settings()` and `options` support in `_compile_kwargs()` so config can control inductor cudagraph behavior.

Why:
- Observed error: “tensor output of CUDAGraphs overwritten by a subsequent run” in teacher/head and block paths.
- CUDAGraphs can reuse/overwrite outputs when multiple compiled calls happen in one step (teacher and student calls). Marking step boundaries mitigates this, and disabling cudagraphs avoids the issue entirely.
- Compile options are exposed so the user can re-enable cudagraphs if they want to experiment with it.

### 4) FSDP wrapper configurability
Files:
- `dinov2/fsdp/__init__.py`
- `dinov2/train/ssl_meta_arch.py`
- `dinov2/configs/ssl_default_config.yaml`
- `dinov2/configs/train/vitg14_reg4.yaml`

What changed:
- `get_fsdp_wrapper(...)` now accepts `fsdp_cfg` and maps config values to:
  - `forward_prefetch`
  - `backward_prefetch`
  - `limit_all_gathers`
  - `sync_module_states`
  - `use_orig_params`
- `SSLMetaArch.prepare_for_distributed_training()` and `LeJEPAMetaArch.prepare_for_distributed_training()` pass `cfg.fsdp`.
- Added `fsdp` block to config defaults and run config.

Why:
- These FSDP knobs materially affect performance and memory behavior; making them config-driven allows fast tuning without code changes.
- Keeping defaults aligned with upstream (prefetch off, limit all-gathers on) avoids accidental regressions.

### 5) Data pipeline tuning knobs (streaming + loader)
Files:
- `dinov2/train/train.py`
- `dinov2/data/loaders.py`
- `dinov2/configs/ssl_default_config.yaml`
- `dinov2/configs/train/vitg14_reg4.yaml`

What changed:
- Streaming dataset builder now accepts:
  - `train.streaming_shuffle_buffer`
  - `train.streaming_fragment_prefetch_limit`
  - `train.streaming_fragment_range_size`
- Dataloader now accepts:
  - `train.dataloader_prefetch_factor`
  - `train.dataloader_persistent_workers`
- `make_data_loader()` now supports `prefetch_factor` and respects `num_workers=0` (no persistent workers/prefetch in that case).

Why:
- The node showed low CPU utilization and IO bottlenecks. These knobs let us raise worker count and prefetch to keep GPUs fed.
- Fragment prefetch and range size are especially relevant for Hugging Face streaming datasets where IO throughput is the limiting factor.

## Debugging Outcomes
- **CUDAGraph overwrite** (teacher head / block path): fixed by marking step begin and then disabling cudagraphs by default. This stabilizes training at the cost of some compile speedup.
- **Performance regression with `torch.compile`**: with cudagraphs disabled, compile overhead can outweigh benefits on this stack. Default is now compile-off in the run config; compile remains available for experimentation.

## Current Config State (vitg14_reg4)
File: `dinov2/configs/train/vitg14_reg4.yaml`

Key performance-related settings after changes:
- `train.compile.enabled: false` (compile disabled by default; can be toggled)
- `compute_precision: bf16` with `grad_scaler: false`
- `fsdp`: defaults aligned to upstream (`forward_prefetch: false`, `limit_all_gathers: true`)
- Data pipeline:
  - `train.num_workers: 8`
  - `train.dataloader_prefetch_factor: 8`
  - `train.dataloader_persistent_workers: true`
  - `train.streaming_fragment_prefetch_limit: 8`

These values were chosen to address IO pressure and keep the GPU side stable; actual optimal values depend on data storage and network bandwidth.

## Tradeoffs and When to Revisit
- **bf16 vs fp16**: H100 supports bf16 well; if you see unexpected slowdowns, consider fp16 + GradScaler (more sensitive to underflow but sometimes faster).
- **Compile**: try `compile_student=true`, `compile_teacher=false`, `compile_heads=false` if you want to test compile on the hottest path only.
- **CUDAGraphs**: re-enable with `train.compile.use_cudagraphs=true` only if you can avoid graph breaks or are comfortable with additional debugging risk.
- **FSDP prefetch**: `forward_prefetch` can help on large networks but may increase memory pressure and reduce overlap if IO is the bottleneck.

## Files Touched (Quick Index)
- `dinov2/train/ssl_meta_arch.py`: compile wrappers, grad scaler gating, compile options, inductor settings.
- `dinov2/train/train.py`: compile step marker, compile hook call, dataloader/streaming knobs.
- `dinov2/fsdp/__init__.py`: FSDP config mapping (prefetch, gathers, sync).
- `dinov2/data/loaders.py`: prefetch_factor support + safe handling of `num_workers=0`.
- `dinov2/configs/ssl_default_config.yaml`: default `fsdp`, `train.compile`, and dataloader/streaming defaults.
- `dinov2/configs/train/vitg14_reg4.yaml`: run-specific overrides for the knobs above.

## Suggested Next Measurements
- Compare `iter_time` and `data_time` in `training_metrics.json` before/after each knob change.
- Profile IO: increase `streaming_fragment_range_size` (e.g., 64MB) if storage supports higher sequential throughput.
- If CPU utilization remains low, consider raising `num_workers` (12–16) and `prefetch_factor` (8–12).
