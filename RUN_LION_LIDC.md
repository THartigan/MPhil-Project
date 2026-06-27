# Running the LION LIDC Checkpoint in Original PaDIS

This duplicate keeps the original PaDIS reconstruction script but adds:

- loading LION `.pt` PaDIS/NCSN++ checkpoints via `lion_checkpoint.py`
- LION-matched LIDC PNG export via `prepare_lidc_pngs.py`
- local `odlstuff` imports instead of the original cluster path

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

For LION `.pt` checkpoints with a sidecar `.json`, `inverse_nodist.py` infers
`--image_size`, `--pad`, `--psize`, and `--channels` from the checkpoint. You can
still pass them explicitly to override the sidecar values.

For CPU-only debugging, use `--device cpu --ct_impl astra_cpu` and reduce
`--steps` and `--max_images` aggressively. The diffusion model is very slow on
CPU, so this is for setup checks rather than paper-quality output.
