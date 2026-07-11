# Running the LION LIDC Checkpoint in Original PaDIS

This duplicate keeps the original PaDIS reconstruction script but adds:

- loading LION `.pt` PaDIS/NCSN++ checkpoints via `lion_checkpoint.py`
- LION-matched LIDC PNG export via `prepare_lidc_pngs.py`
- local `odlstuff` imports instead of the original cluster path
- optional `--sampler dps|pc|langevin|ddnm|patch_average|patch_stitch` selection for diagnostics against
  the public helper functions; the default remains the README DPS path
- optional `--patch_batch_size` microbatching for the public helper denoiser
- optional `--checkpoint_denoiser` for `patch_average`/`patch_stitch` DPS
  helper runs that need denoiser gradients on smaller GPUs

## 1. Export LIDC images with LION preprocessing

The default export path uses `LION.data_loaders.LIDC_IDRI` with `task="image_prior"`.
That matches the training script for `padis_lidc_256.pt`: raw HU slices are resized
to `256x256` with bilinear interpolation and `align_corners=False`, then transformed
with `from_HU_to_normal`, i.e. `(HU + 1000) / 3000` clipped to `[0, 1]`.

```bash
conda run -n padis-dev env PYTHONPATH=/home/thomas/DiS/Project/LION \
  python prepare_lidc_pngs.py \
  --split test \
  --input-root /home/thomas/DiS/Project/Data/processed/LIDC-IDRI \
  --output-dir /home/thomas/DiS/Project/Data/processed/LIDC-IDRI-padis-png-256 \
  --image-size 256
```

For a quick smoke run, add `--limit 1`.

## 2. Run original PaDIS reconstruction with the LION checkpoint

On a GPU node, use the paper-style CUDA/ASTRA backend:

```bash
conda run -n padis-dev env PYTHONPATH=/home/thomas/DiS/Project/LION \
  MPLCONFIGDIR=/tmp/padis-mpl XDG_CACHE_HOME=/tmp/padis-xdg \
  python inverse_nodist.py \
  --network /home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/padis_lidc_256_default_10h_local_20260624_232630/patch_lidc_default_10h_local/padis_lidc_256.pt \
  --lion_repo /home/thomas/DiS/Project/LION \
  --device cuda \
  --ct_impl astra_cuda \
  --image_dir /home/thomas/DiS/Project/Data/processed/LIDC-IDRI-padis-png-256 \
  --outdir /home/thomas/DiS/Project/Data/experiments/PaDIS/original_repo_lion_lidc_256 \
  --name ct_parbeam \
  --views 20 \
  --steps 100 \
  --sigma_min 0.003 \
  --sigma_max 10 \
  --zeta 0.3 \
  --sigma 0 \
  --intermediate_interval 5 \
  --max_images 1
```

Outputs are written as per-image PNG reconstructions plus `reconstructions.npz`
containing `clean`, `recon`, `psnr`, `ssim`, and `files`.
Intermediate sampler PNGs are written to `OUTDIR/intermediates/<sample-name>/`
by default; set `--intermediate_interval 0` to disable them.

## Quick trace alignment against LION geometry

The added `ct_lion_fanbeam` and `ct_lion_parbeam` modes use the LION-scale
geometry for comparison with the LION-native reconstruction path. Their ODL
adjoint has a different numeric scale from the public README `ct_parbeam`
geometry, so these modes apply `data_gradient_scale=0.09` by default inside
the DPS norm-gradient step. The original `ct_parbeam` path remains unchanged.

The public README command executes DPS. The `--sampler` option is a diagnostic
extension in this LION-compatible fork that exposes helper functions already
present in the public script:

| Sampler | Public helper |
|---|---|
| `dps` | `dps(...)`, the README/default reconstruction path |
| `pc` | `pc_sampling(...)` |
| `langevin` | `langevin(..., ddnm=False)` |
| `ddnm` | `langevin(..., ddnm=True)` |
| `patch_average` | `denoisedOverlap(...)` inside the DPS loop |
| `patch_stitch` | `denoisedTile(...)` inside the DPS loop |

Use `--patch_batch_size 1` for helper samplers on 8 GB GPUs. `pc`, `langevin`,
and `ddnm` do not need denoiser backpropagation, so the fork runs their patch
denoising under `torch.no_grad()`. DPS, `patch_average`, and `patch_stitch`
keep denoiser gradients because their norm-gradient data-consistency step
differentiates through the denoiser. Add `--checkpoint_denoiser` for
`dps`, `patch_average`, or `patch_stitch` on 8 GB GPUs.

For 512 checkpoints, the public DPS path also needs the image width passed into
the measurement-gradient crop. The upstream helper default is `w=256`; leaving
that default active makes a 512 reconstruction compare a 256 crop against the
512 projection operator. This fork passes the inferred image width through the
public DPS and fixed-patch DPS calls. It also detaches `x_next` between inner
DPS updates; this preserves the update rule but prevents PyTorch from retaining
the full 100-step graph.

Observed 512 public-reference run:

```bash
conda run --no-capture-output -n padis-dev env PYTHONPATH=/home/thomas/DiS/Project/LION \
  MPLCONFIGDIR=/tmp/padis-mpl XDG_CACHE_HOME=/tmp/padis-xdg \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python inverse_nodist.py \
  --network /home/thomas/DiS/Project/Data/experiments/PaDIS/external_models/padis_lidc_512.pt \
  --lion_repo /home/thomas/DiS/Project/LION \
  --device cuda \
  --ct_impl astra_cuda \
  --image_dir /home/thomas/DiS/Project/Data/processed/LIDC-IDRI-padis-png-512-validation-start4-limit1 \
  --outdir /home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/public_repo_fork_512_parbeam_60v_full_detach_checkpoint_20260708 \
  --name ct_parbeam \
  --views 60 \
  --steps 100 \
  --sigma_min 0.002 \
  --sigma_max 10 \
  --rho 7 \
  --zeta 0.3 \
  --sigma 0 \
  --patch_batch_size 1 \
  --checkpoint_denoiser \
  --intermediate_interval 20 \
  --trace_interval 20 \
  --max_images 1 \
  --seed 33
```

This completed on the local GTX 1070 in 1:56:41 for one 512 validation slice.
The public FBP initialization reached PSNR 29.19 dB and SSIM 0.706. The final
public DPS reconstruction reached PSNR 32.61 dB and SSIM 0.802, with expected
high-noise intermediates at steps 0-40, anatomical recovery around step 60, and
near-final recovery by step 80.

The paper describes geometrically spaced CT noise levels, while the original
README command uses the EDM/Karras-style `rho=7` spacing. This fork therefore
also exposes `--sigma_schedule geometric` for paper-schedule checks while
leaving the default README behavior unchanged. On the same 512 validation slice,
the paper-schedule public fork run was:

```bash
conda run --no-capture-output -n padis-dev env PYTHONPATH=/home/thomas/DiS/Project/LION \
  MPLCONFIGDIR=/tmp/padis-mpl XDG_CACHE_HOME=/tmp/padis-xdg \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python inverse_nodist.py \
  --network /home/thomas/DiS/Project/Data/experiments/PaDIS/external_models/padis_lidc_512.pt \
  --lion_repo /home/thomas/DiS/Project/LION \
  --device cuda \
  --ct_impl astra_cuda \
  --image_dir /home/thomas/DiS/Project/Data/processed/LIDC-IDRI-padis-png-512-validation-start4-limit1 \
  --outdir /home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/public_repo_fork_512_parbeam60_geometric_zeta03_20260708 \
  --name ct_parbeam \
  --views 60 \
  --steps 100 \
  --sigma_min 0.002 \
  --sigma_max 10 \
  --rho 7 \
  --sigma_schedule geometric \
  --zeta 0.3 \
  --sigma 0 \
  --patch_batch_size 1 \
  --checkpoint_denoiser \
  --intermediate_interval 20 \
  --trace_interval 20 \
  --max_images 1 \
  --seed 33
```

This completed on the local GTX 1070 in about 1:59:00 for one 512 validation
slice. The final public DPS reconstruction reached PSNR 33.81 dB and SSIM
0.807. This is the current single-slice public-fork anchor for LION
`--implementation public_repo` 512/60 tuning.

To isolate the effect of the LION CT operator from the public sampler itself,
the same public DPS path was also run through the fork's `ct_lion_fanbeam`
shim:

```bash
conda run --no-capture-output -n padis-dev env PYTHONPATH=/home/thomas/DiS/Project/LION \
  MPLCONFIGDIR=/tmp/padis-mpl XDG_CACHE_HOME=/tmp/padis-xdg \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python inverse_nodist.py \
  --network /home/thomas/DiS/Project/Data/experiments/PaDIS/external_models/padis_lidc_512.pt \
  --lion_repo /home/thomas/DiS/Project/LION \
  --device cuda \
  --image_dir /home/thomas/DiS/Project/Data/processed/LIDC-IDRI-padis-png-512-validation-start4-limit1 \
  --outdir /home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/public_repo_fork_512_lion_fanbeam60_geometric_zeta03_full_20260709 \
  --name ct_lion_fanbeam \
  --views 60 \
  --steps 100 \
  --sigma_min 0.002 \
  --sigma_max 10 \
  --rho 7 \
  --sigma_schedule geometric \
  --zeta 0.3 \
  --sigma 0 \
  --patch_batch_size 1 \
  --checkpoint_denoiser \
  --intermediate_interval 0 \
  --trace_interval 20 \
  --max_images 1 \
  --seed 33
```

This completed on the local GTX 1070 in 1:56:02 for the same validation slice
and reached PSNR 33.17 dB and SSIM 0.804. The LION-fanbeam public-fork anchor
therefore sits below the public `ct_parbeam` anchor, which means the remaining
512/60 gap to the parallel-beam public run is largely a CT operator/geometry
comparison rather than a failure of the LION public-compatible sampler.

The upstream `denoisedOverlap(...)` helper overruns the padded image for the
README defaults (`image_size=256`, `pad=24`, `psize=56`, `overlap=8`). This fork
keeps the public helper semantics but clips that final overlap start to the last
valid patch when the helper is reached through `--sampler patch_average`.
`patch_stitch` keeps the public helper's hard-coded start index `4`.

For fast debugging, stop after the first outer sampler step while preserving
the full 100-step EDM schedule:

```bash
conda run --no-capture-output -n padis-dev env PYTHONPATH=/home/thomas/DiS/Project/LION \
  MPLCONFIGDIR=/tmp/padis-mpl XDG_CACHE_HOME=/tmp/padis-xdg \
  python inverse_nodist.py \
  --network /home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/padis_lidc_256_default_10h_local_20260624_232630/patch_lidc_default_10h_local/padis_lidc_256.pt \
  --lion_repo /home/thomas/DiS/Project/LION \
  --device cuda \
  --ct_impl astra_cuda \
  --image_dir /home/thomas/DiS/Project/Data/processed/LIDC-IDRI-padis-png-256 \
  --outdir /home/thomas/DiS/Project/Data/experiments/PaDIS/trace_alignment_public_lion_fanbeam_1step \
  --name ct_lion_fanbeam \
  --views 20 \
  --steps 100 \
  --sigma_min 0.003 \
  --sigma_max 10 \
  --zeta 0.3 \
  --sigma 0 \
  --intermediate_interval 1 \
  --trace_interval 1 \
  --stop_after_outer_steps 1 \
  --max_images 1 \
  --seed 2
```

To smoke-test the public predictor-corrector helper on the same LION geometry:

```bash
conda run --no-capture-output -n padis-dev env PYTHONPATH=/home/thomas/DiS/Project/LION \
  MPLCONFIGDIR=/tmp/padis-mpl XDG_CACHE_HOME=/tmp/padis-xdg \
  python inverse_nodist.py \
  --network /home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/padis_lidc_256_default_10h_local_20260624_232630/patch_lidc_default_10h_local/padis_lidc_256.pt \
  --lion_repo /home/thomas/DiS/Project/LION \
  --device cuda \
  --ct_impl astra_cuda \
  --image_dir /home/thomas/DiS/Project/Data/processed/LIDC-IDRI-padis-png-256 \
  --outdir /home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_public_helper_sampler_pc_smoke_20260628 \
  --name ct_lion_fanbeam \
  --views 20 \
  --steps 100 \
  --sigma_min 0.002 \
  --sigma_max 10 \
  --rho 7 \
  --zeta 0.3 \
  --sigma 0 \
  --sampler pc \
  --patch_batch_size 1 \
  --intermediate_interval 1 \
  --trace_interval 0 \
  --stop_after_outer_steps 1 \
  --max_images 1 \
  --seed 2
```

To smoke-test the public patch averaging helper, use:

```bash
conda run --no-capture-output -n padis-dev env PYTHONPATH=/home/thomas/DiS/Project/LION \
  MPLCONFIGDIR=/tmp/padis-mpl XDG_CACHE_HOME=/tmp/padis-xdg \
  python inverse_nodist.py \
  --network /home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/padis_lidc_256_default_10h_local_20260624_232630/patch_lidc_default_10h_local/padis_lidc_256.pt \
  --lion_repo /home/thomas/DiS/Project/LION \
  --device cuda \
  --ct_impl astra_cuda \
  --image_dir /home/thomas/DiS/Project/Data/processed/LIDC-IDRI-padis-png-256 \
  --outdir /home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/public_helper_patch_average_smoke \
  --name ct_lion_fanbeam \
  --views 20 \
  --steps 100 \
  --sigma_min 0.002 \
  --sigma_max 10 \
  --rho 7 \
  --zeta 0.3 \
  --sigma 0 \
  --sampler patch_average \
  --patch_batch_size 1 \
  --checkpoint_denoiser \
  --intermediate_interval 1 \
  --trace_interval 1 \
  --stop_after_outer_steps 1 \
  --max_images 1 \
  --seed 2
```

Use `--sampler patch_stitch` and a different `--outdir` for the stitching
helper.

For LION `.pt` checkpoints with a sidecar `.json`, `inverse_nodist.py` infers
`--image_size`, `--pad`, `--psize`, and `--channels` from the checkpoint. You can
still pass them explicitly to override the sidecar values.

For CPU-only debugging, use `--device cpu --ct_impl astra_cpu` and reduce
`--steps` and `--max_images` aggressively. The diffusion model is very slow on
CPU, so this is for setup checks rather than paper-quality output.
