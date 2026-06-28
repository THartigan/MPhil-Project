import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from training.pos_embedding import Pos_Embedding
import scipy.io
import json
import random
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import sys
from inverse_operators import *
from denoise_padding import denoisedFromPatches, getIndices, denoisedOverlap, denoisedTile
from lion_checkpoint import load_lion_padis_model, read_lion_padis_metadata

def log(message):
    print(message, flush=True)

def tensor_stats(prefix, tensor):
    tensor = tensor.detach()
    return {
        f'{prefix}_min': float(tensor.amin().cpu()),
        f'{prefix}_max': float(tensor.amax().cpu()),
        f'{prefix}_mean': float(tensor.mean().cpu()),
        f'{prefix}_std': float(tensor.std().cpu()),
        f'{prefix}_norm': float(torch.linalg.norm(tensor).cpu()),
    }

def set_run_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def makeFigures(noisy2, denoised2, orig2, i, imsize=256, preview_dir=None):
    preview_dir = preview_dir or os.environ.get('PADIS_INTERMEDIATE_DIR')
    if not preview_dir:
        return
    channels = len(denoised2[:,0,0])
    os.makedirs(preview_dir, exist_ok=True)
    denoised = torch.clone(denoised2)
    noisy = torch.clone(noisy2)
    orig = orig2.copy()
    orig = np.transpose(orig, (1,2,0))

    denoised = torch.squeeze(denoised).cpu().numpy()
    orig = np.squeeze(orig)
    noisy = torch.squeeze(noisy).cpu().numpy()

    if channels > 1:
        noisy = np.transpose(noisy, (1,2,0))
        denoised = np.transpose(denoised, (1,2,0))

    noisy = np.clip(noisy, 0, 1)
    denoised = np.clip(denoised, 0,1)
    orig = np.clip(orig, 0, 1)

    plt.imsave(
        os.path.join(preview_dir, f'{i}_recon.png'),
        denoised,
        cmap='gray',
        vmin=0,
        vmax=1,
    )
    plt.imsave(
        os.path.join(preview_dir, f'{i}_fbp.png'),
        noisy,
        cmap='gray',
        vmin=0,
        vmax=1,
    )
    np.save(os.path.join(preview_dir, f'{i}_recon.npy'), denoised)

    noisypsnr = psnr(noisy, orig, data_range=1)
    denoisedpsnr = psnr(denoised, orig, data_range=1)
    t1 = 'FBP recon'
    t2 = 'Diffusion recon'

    plt.figure(figsize=(12,6))
    plt.subplot(1,3,1),plt.imshow(noisy, cmap='gray'),plt.axis('off'),plt.title(str(noisypsnr))
    plt.subplot(1,3,2),plt.imshow(denoised, cmap='gray'),plt.axis('off'),plt.title(str(denoisedpsnr))
    plt.subplot(1,3,3),plt.imshow(orig, cmap='gray'),plt.axis('off')

    plt.savefig(os.path.join(preview_dir, f'{i}.png'))
    plt.close('all')

def image_array(tensor):
    array = torch.squeeze(tensor.detach()).cpu().numpy()
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    return np.clip(array, 0, 1)

def saveTraceSnapshot(
    preview_dir,
    step,
    inner,
    pad,
    w,
    *,
    x,
    denoised,
    projected,
    x_next,
    algorithm='public_dps',
):
    if not preview_dir:
        return
    folder = os.path.join(preview_dir, 'trace_images')
    os.makedirs(folder, exist_ok=True)
    stem = f'step_{step:04d}_inner_{inner:02d}_{algorithm}'
    cropped = {
        'current': image_array(x[:, pad:pad+w, pad:pad+w]),
        'denoised': image_array(denoised[:, pad:pad+w, pad:pad+w]),
        'projected': image_array(projected[:, pad:pad+w, pad:pad+w]),
        'x_next': image_array(x_next[:, pad:pad+w, pad:pad+w]),
    }
    np.savez_compressed(os.path.join(folder, f'{stem}.npz'), **cropped)
    for name, array in cropped.items():
        plt.imsave(
            os.path.join(folder, f'{stem}_{name}.png'),
            array,
            cmap='gray',
            vmin=0,
            vmax=1,
        )

def pinv(net, latents, latents_pos, inverseop, noisy=None, randn_like = torch.randn_like, num_steps=18,
              clean=None, sigma_min=0.005, sigma_max = 0.05, rho=7, zeta=0.3, pad=64, psize=64,
              S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,):
    w = len(latents[0,0,0,:])
    fbp = torch.clamp(inverseop.Adagger(noisy), min=0, max=1)
    return torch.nn.functional.pad(fbp, (pad, pad, pad, pad), "constant", 0)

def validate_stop_after_outer_steps(stop_after_outer_steps):
    if stop_after_outer_steps is None:
        return None
    stop_after_outer_steps = int(stop_after_outer_steps)
    if stop_after_outer_steps <= 0:
        raise ValueError('stop_after_outer_steps must be positive or None')
    return stop_after_outer_steps

def pc_sampling(net, latents, latents_pos, inverseop, noisy=None, randn_like = torch.randn_like, num_steps=18,
              clean=None, sigma_min=0.005, sigma_max = 0.05, rho=7, zeta=0.3, pad=64, psize=64,
              S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
              intermediate_dir=None, intermediate_interval=5, stop_after_outer_steps=None,
              patch_batch_size=None, trace_file=None, trace_interval=0,):
    w = len(latents[0,0,0,:])
    patches = w // psize + 1
    spaced = np.linspace(0, (patches-1)*psize, patches, dtype=int)
    x_init = torch.clamp(inverseop.Adagger(noisy), min=0, max=1)
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    x_init = sigma_max * torch.randn_like(x_init)
    x = torch.nn.functional.pad(x_init, (pad, pad, pad, pad), "constant", 0)
    stop_after_outer_steps = validate_stop_after_outer_steps(stop_after_outer_steps)
    trace_records = []
    measurement_norm = torch.linalg.norm(noisy.detach()).clamp_min(1e-12)

    for i, (t_cur, t_next) in tqdm.tqdm(
        enumerate(zip(t_steps[:-1], t_steps[1:])),
        total=min(num_steps, stop_after_outer_steps) if stop_after_outer_steps is not None else num_steps,
        desc='PaDIS PC outer steps',
        dynamic_ncols=True,
        file=sys.stdout,
    ):
        if stop_after_outer_steps is not None and i >= stop_after_outer_steps:
            break
        if i == num_steps-1:
            break
        indices = getIndices(spaced, patches, pad, psize)
        D = denoisedFromPatches(net, torch.unsqueeze(x, 0), t_cur, latents_pos, None, indices, t_goal=0, wrong=False, batch_size=patch_batch_size, track_grad=False)
        D = torch.squeeze(D, dim=0)
        score = (D-x)/t_cur**2
        x = x + (t_cur**2 - t_next**2) * score
        z = randn_like(x)
        x = x + torch.sqrt(t_cur**2 - t_next**2) * z #predictor step
        x_predictor = x.clone()

        Ainput = noisy-inverseop.A(x[:,pad:pad+w, pad:pad+w].to(dtype=torch.float32))
        stepsize = zeta/torch.norm(Ainput)
        raw_correction = inverseop.AT(Ainput)
        correction = stepsize * raw_correction
        x[:,pad:pad+w, pad:pad+w] += correction
        should_trace = (
            trace_interval > 0
            and (i % trace_interval == 0 or i == num_steps - 1)
        )
        if should_trace:
            item = {
                'algorithm': 'public_pc_predictor',
                'step': int(i),
                'inner': 0,
                'sigma': float(t_cur.detach().cpu()),
                'next_sigma': float(t_next.detach().cpu()),
                'predictor_delta': float((t_cur**2 - t_next**2).detach().cpu()),
                'data_step_size': float(stepsize.detach().cpu()),
                'residual_norm': float(torch.linalg.norm(Ainput.detach()).cpu()),
                'measurement_norm': float(measurement_norm.detach().cpu()),
                'relative_residual_norm': float((torch.linalg.norm(Ainput.detach()) / measurement_norm).cpu()),
                'score_norm': float(torch.linalg.norm(score.detach()).cpu()),
                'z_norm': float(torch.linalg.norm(z.detach()).cpu()),
                'raw_gradient_norm': float(torch.linalg.norm(raw_correction.detach()).cpu()),
                'gradient_norm': float(torch.linalg.norm(correction.detach()).cpu()),
            }
            item.update(tensor_stats('x_before_data', x_predictor))
            item.update(tensor_stats('x', x))
            item.update(tensor_stats('denoised', D))
            item.update(tensor_stats('residual', Ainput))
            trace_records.append(item)

        if i < num_steps-1:
            z = randn_like(x)
            D = denoisedFromPatches(net, torch.unsqueeze(x, 0), t_cur, latents_pos, None, indices, t_goal=0, wrong=False, batch_size=patch_batch_size, track_grad=False)
            D = torch.squeeze(D, dim=0)
            score = (D-x)/t_next**2
            r = 0.16
            eps = 2*r*torch.norm(z)/torch.norm(score)
            x = x + eps * score
            x = x + torch.sqrt(2*eps)*z #corrector step
            x_corrector = x.clone()

            Ainput = noisy-inverseop.A(x[:,pad:pad+w, pad:pad+w].to(dtype=torch.float32))
            stepsize = zeta/torch.norm(Ainput)* min(40, t_cur*200)
            raw_correction = inverseop.AT(Ainput)
            correction = stepsize * raw_correction
            x[:,pad:pad+w, pad:pad+w] += correction
            if should_trace:
                item = {
                    'algorithm': 'public_pc_corrector',
                    'step': int(i),
                    'inner': 1,
                    'sigma': float(t_cur.detach().cpu()),
                    'next_sigma': float(t_next.detach().cpu()),
                    'eps': float(eps.detach().cpu()),
                    'data_step_size': float(stepsize.detach().cpu()),
                    'residual_norm': float(torch.linalg.norm(Ainput.detach()).cpu()),
                    'measurement_norm': float(measurement_norm.detach().cpu()),
                    'relative_residual_norm': float((torch.linalg.norm(Ainput.detach()) / measurement_norm).cpu()),
                    'score_norm': float(torch.linalg.norm(score.detach()).cpu()),
                    'z_norm': float(torch.linalg.norm(z.detach()).cpu()),
                    'raw_gradient_norm': float(torch.linalg.norm(raw_correction.detach()).cpu()),
                    'gradient_norm': float(torch.linalg.norm(correction.detach()).cpu()),
                }
                item.update(tensor_stats('x_before_data', x_corrector))
                item.update(tensor_stats('x', x))
                item.update(tensor_stats('denoised', D))
                item.update(tensor_stats('residual', Ainput))
                trace_records.append(item)

        if intermediate_dir and intermediate_interval > 0:
            if i % intermediate_interval == 0 or i == num_steps-1:
                makeFigures(
                    x_init,
                    x[:,pad:pad+w, pad:pad+w].detach(),
                    clean,
                    i,
                    preview_dir=intermediate_dir,
                )
    if trace_file and trace_interval > 0:
        os.makedirs(os.path.dirname(trace_file) or '.', exist_ok=True)
        with open(trace_file, 'w') as f:
            json.dump([{'index': 0, 'trace': trace_records}], f, indent=2)
    return x

def measurement_cond_fn(
    measurement,
    x_prev,
    x0hat,
    inverseop,
    pad=24,
    w=256,
    return_details=False,
    data_gradient_scale=None,
):
    difference = measurement - inverseop.A(x0hat[:,pad:pad+w, pad:pad+w]).to(dtype=torch.float32)
    norm = torch.linalg.norm(difference)
    # Using gradient of norm instead of norm^2 is equivalent (within a factor of 2)
    # of using gradient of norm^2 and then using a "normalized" step size
    # as recommended in Footnote 5 and Appendix D.1 of the 2023 ICLR paper on DPS.
    raw_norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
    if data_gradient_scale is None:
        data_gradient_scale = getattr(inverseop, 'data_gradient_scale', 1.0)
    data_gradient_scale = float(data_gradient_scale)
    norm_grad = raw_norm_grad * data_gradient_scale
    if return_details:
        return norm_grad, raw_norm_grad, difference, norm, data_gradient_scale
    return norm_grad

def dps(net, latents, latents_pos, inverseop, noisy=None, randn_like = torch.randn_like, num_steps=18,
              clean=None, sigma_min=0.005, sigma_max = 0.05, rho=7, zeta=0.3, pad=64, psize=64,
              S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
              intermediate_dir=None, intermediate_interval=5, data_gradient_scale=None,
              trace_file=None, trace_interval=0, stop_after_outer_steps=None,
              patch_batch_size=None,):
    w = len(latents[0,0,0,:])
    patches = w // psize + 1
    spaced = np.linspace(0, (patches-1)*psize, patches, dtype=int)
    x_init = torch.clamp(inverseop.Adagger(noisy), min=0, max=1)
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    x = sigma_max * torch.randn_like(x_init)
    x = torch.nn.functional.pad(x_init, (pad, pad, pad, pad), "constant", 0).requires_grad_()
    if intermediate_dir and intermediate_interval > 0:
        makeFigures(
            x_init,
            x[:, pad:pad+w, pad:pad+w].detach(),
            clean,
            'initial',
            preview_dir=intermediate_dir,
        )
    trace_records = []
    measurement_norm = torch.linalg.norm(noisy.detach()).clamp_min(1e-12)
    if data_gradient_scale is None:
        data_gradient_scale = getattr(inverseop, 'data_gradient_scale', 1.0)
    data_gradient_scale = float(data_gradient_scale)
    stop_after_outer_steps = validate_stop_after_outer_steps(stop_after_outer_steps)
    for i, (t_cur, t_next) in tqdm.tqdm(
        enumerate(zip(t_steps[:-1], t_steps[1:])),
        total=min(num_steps, stop_after_outer_steps) if stop_after_outer_steps is not None else num_steps,
        desc='PaDIS DPS outer steps',
        dynamic_ncols=True,
        file=sys.stdout,
    ):
        if stop_after_outer_steps is not None and i >= stop_after_outer_steps:
            break
        alpha = 0.5*t_cur**2
        for j in range(10):
            indices = getIndices(spaced, patches, pad, psize)
            D = denoisedFromPatches(net, torch.unsqueeze(x, 0), t_cur, latents_pos, None, indices, t_goal=0, wrong=False, batch_size=patch_batch_size, track_grad=True)
            D = torch.squeeze(D, dim=0)
            score = (D-x)/t_cur**2
            z = randn_like(x)

            x0hat = D
            should_trace = (
                trace_interval > 0
                and (i % trace_interval == 0 or i == num_steps - 1)
                and (j == 0 or j == 9)
            )
            if should_trace:
                norm_grad, raw_norm_grad, difference, residual_norm, effective_data_gradient_scale = measurement_cond_fn(
                    noisy,
                    x,
                    x0hat,
                    inverseop,
                    pad=pad,
                    return_details=True,
                    data_gradient_scale=data_gradient_scale,
                )
            else:
                norm_grad = measurement_cond_fn(
                    noisy,
                    x,
                    x0hat,
                    inverseop,
                    pad=pad,
                    data_gradient_scale=data_gradient_scale,
                )
                raw_norm_grad = None
                difference = None
                residual_norm = None
                effective_data_gradient_scale = data_gradient_scale

            x_after_data = x - zeta * norm_grad

            if i < num_steps - 1:
                x_next = x_after_data + alpha/2 * score + torch.sqrt(alpha) * z
            else:
                x_next = x_after_data + alpha/2 * score
            if should_trace:
                saveTraceSnapshot(
                    intermediate_dir,
                    i,
                    j,
                    pad,
                    w,
                    x=x,
                    denoised=D,
                    projected=x_after_data,
                    x_next=x_next,
                )
                item = {
                    'algorithm': 'public_dps',
                    'step': int(i),
                    'inner': int(j),
                    'sigma': float(t_cur.detach().cpu()),
                    'alpha': float(alpha.detach().cpu()),
                    'residual_norm': float(residual_norm.detach().cpu()),
                    'measurement_norm': float(measurement_norm.detach().cpu()),
                    'relative_residual_norm': float((residual_norm / measurement_norm).detach().cpu()),
                    'score_norm': float(torch.linalg.norm(score.detach()).cpu()),
                    'gradient_norm': float(torch.linalg.norm(norm_grad.detach()).cpu()),
                    'raw_gradient_norm': float(torch.linalg.norm(raw_norm_grad.detach()).cpu()),
                    'data_gradient_scale': float(effective_data_gradient_scale),
                    'z_norm': float(torch.linalg.norm(z.detach()).cpu()),
                }
                item.update(tensor_stats('x', x))
                item.update(tensor_stats('denoised', D))
                item.update(tensor_stats('residual', difference))
                item.update(tensor_stats('projected', x_after_data))
                item.update(tensor_stats('x_next', x_next))
                trace_records.append(item)
            x = x_next
        if intermediate_dir and intermediate_interval > 0:
            if i % intermediate_interval == 0 or i == num_steps-1:
                makeFigures(
                    x_init,
                    x[:,pad:pad+w, pad:pad+w].detach(),
                    clean,
                    f'step_{i:04d}',
                    preview_dir=intermediate_dir,
                )
    if trace_file and trace_interval > 0:
        os.makedirs(os.path.dirname(trace_file) or '.', exist_ok=True)
        with open(trace_file, 'w') as f:
            json.dump([{'index': 0, 'trace': trace_records}], f, indent=2)
    return x.detach()


def fixed_patch_dps(net, latents, latents_pos, inverseop, noisy=None, randn_like = torch.randn_like, num_steps=18,
              clean=None, sigma_min=0.005, sigma_max = 0.05, rho=7, zeta=0.3, pad=64, psize=64,
              S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
              intermediate_dir=None, intermediate_interval=5, data_gradient_scale=None,
              trace_file=None, trace_interval=0, stop_after_outer_steps=None,
              patch_batch_size=None, sampler='patch_average', overlap=8,
              checkpoint_denoiser=False):
    del S_churn, S_min, S_max, S_noise
    if sampler == 'patch_average':
        denoise_fn = denoisedOverlap
        algorithm_name = 'public_patch_average'
    elif sampler == 'patch_stitch':
        denoise_fn = denoisedTile
        algorithm_name = 'public_patch_stitch'
    else:
        raise ValueError(f'Unknown fixed patch sampler: {sampler}')

    w = len(latents[0,0,0,:])
    x_init = torch.clamp(inverseop.Adagger(noisy), min=0, max=1)
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    x = sigma_max * torch.randn_like(x_init)
    x = torch.nn.functional.pad(x_init, (pad, pad, pad, pad), "constant", 0).requires_grad_()
    if intermediate_dir and intermediate_interval > 0:
        makeFigures(
            x_init,
            x[:, pad:pad+w, pad:pad+w].detach(),
            clean,
            'initial',
            preview_dir=intermediate_dir,
        )
    trace_records = []
    measurement_norm = torch.linalg.norm(noisy.detach()).clamp_min(1e-12)
    if data_gradient_scale is None:
        data_gradient_scale = getattr(inverseop, 'data_gradient_scale', 1.0)
    data_gradient_scale = float(data_gradient_scale)
    stop_after_outer_steps = validate_stop_after_outer_steps(stop_after_outer_steps)
    for i, (t_cur, t_next) in tqdm.tqdm(
        enumerate(zip(t_steps[:-1], t_steps[1:])),
        total=min(num_steps, stop_after_outer_steps) if stop_after_outer_steps is not None else num_steps,
        desc=f'PaDIS {sampler} outer steps',
        dynamic_ncols=True,
        file=sys.stdout,
    ):
        if stop_after_outer_steps is not None and i >= stop_after_outer_steps:
            break
        alpha = 0.5*t_cur**2
        for j in range(10):
            D = denoise_fn(
                net,
                torch.unsqueeze(x, 0),
                t_cur,
                latents_pos,
                None,
                pad=pad,
                psize=psize,
                overlap=overlap,
                t_goal=0,
                batch_size=patch_batch_size,
                track_grad=True,
                safe_bounds=True,
                use_checkpoint=checkpoint_denoiser,
            )
            D = torch.squeeze(D, dim=0)
            score = (D-x)/t_cur**2
            z = randn_like(x)

            x0hat = D
            should_trace = (
                trace_interval > 0
                and (i % trace_interval == 0 or i == num_steps - 1)
                and (j == 0 or j == 9)
            )
            if should_trace:
                norm_grad, raw_norm_grad, difference, residual_norm, effective_data_gradient_scale = measurement_cond_fn(
                    noisy,
                    x,
                    x0hat,
                    inverseop,
                    pad=pad,
                    return_details=True,
                    data_gradient_scale=data_gradient_scale,
                )
            else:
                norm_grad = measurement_cond_fn(
                    noisy,
                    x,
                    x0hat,
                    inverseop,
                    pad=pad,
                    data_gradient_scale=data_gradient_scale,
                )
                raw_norm_grad = None
                difference = None
                residual_norm = None
                effective_data_gradient_scale = data_gradient_scale

            x_after_data = x - zeta * norm_grad

            if i < num_steps - 1:
                x_next = x_after_data + alpha/2 * score + torch.sqrt(alpha) * z
            else:
                x_next = x_after_data + alpha/2 * score
            if should_trace:
                saveTraceSnapshot(
                    intermediate_dir,
                    i,
                    j,
                    pad,
                    w,
                    x=x,
                    denoised=D,
                    projected=x_after_data,
                    x_next=x_next,
                    algorithm=algorithm_name,
                )
                item = {
                    'algorithm': algorithm_name,
                    'step': int(i),
                    'inner': int(j),
                    'sigma': float(t_cur.detach().cpu()),
                    'alpha': float(alpha.detach().cpu()),
                    'residual_norm': float(residual_norm.detach().cpu()),
                    'measurement_norm': float(measurement_norm.detach().cpu()),
                    'relative_residual_norm': float((residual_norm / measurement_norm).detach().cpu()),
                    'score_norm': float(torch.linalg.norm(score.detach()).cpu()),
                    'gradient_norm': float(torch.linalg.norm(norm_grad.detach()).cpu()),
                    'raw_gradient_norm': float(torch.linalg.norm(raw_norm_grad.detach()).cpu()),
                    'data_gradient_scale': float(effective_data_gradient_scale),
                    'z_norm': float(torch.linalg.norm(z.detach()).cpu()),
                    'patch_overlap': int(overlap),
                }
                item.update(tensor_stats('x', x))
                item.update(tensor_stats('denoised', D))
                item.update(tensor_stats('residual', difference))
                item.update(tensor_stats('projected', x_after_data))
                item.update(tensor_stats('x_next', x_next))
                trace_records.append(item)
            x = x_next
        if intermediate_dir and intermediate_interval > 0:
            if i % intermediate_interval == 0 or i == num_steps-1:
                makeFigures(
                    x_init,
                    x[:,pad:pad+w, pad:pad+w].detach(),
                    clean,
                    f'step_{i:04d}',
                    preview_dir=intermediate_dir,
                )
    if trace_file and trace_interval > 0:
        os.makedirs(os.path.dirname(trace_file) or '.', exist_ok=True)
        with open(trace_file, 'w') as f:
            json.dump([{'index': 0, 'trace': trace_records}], f, indent=2)
    return x.detach()

def langevin(net, latents, latents_pos, inverseop, noisy=None, randn_like = torch.randn_like, num_steps=18,
              clean=None, sigma_min=0.005, sigma_max = 0.05, rho=7, zeta=0.3, pad=64, psize=64,
              S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, ddnm=False,
              intermediate_dir=None, intermediate_interval=5, stop_after_outer_steps=None,
              patch_batch_size=None):
    w = len(latents[0,0,0,:])
    x_init = torch.clamp(inverseop.Adagger(noisy), min=0, max=1)

    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    #t_steps = torch.from_numpy(np.geomspace(sigma_max, sigma_min, num=num_steps)).to(latents.device)
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
    #print(t_steps)
    x = x_init #might initialize with pure noise, depends
    x = torch.nn.functional.pad(x, (pad, pad, pad, pad), "constant", 0)
    x = sigma_max * torch.randn_like(x)

    patches = w // psize + 1
    spaced = np.linspace(0, (patches-1)*psize, patches, dtype=int)
    stop_after_outer_steps = validate_stop_after_outer_steps(stop_after_outer_steps)
    #print(x.shape)
    for i, (t_cur, t_next) in tqdm.tqdm(
        enumerate(zip(t_steps[:-1], t_steps[1:])),
        total=min(num_steps, stop_after_outer_steps) if stop_after_outer_steps is not None else num_steps,
        desc='PaDIS Langevin outer steps',
        dynamic_ncols=True,
        file=sys.stdout,
    ):
        if stop_after_outer_steps is not None and i >= stop_after_outer_steps:
            break
        alpha = 1*t_cur**2
        for j in range(10):
            indices = getIndices(spaced, patches, pad, psize)
            D = denoisedFromPatches(net, torch.unsqueeze(x, 0), t_cur, latents_pos, None, indices, t_goal=0, wrong=False, batch_size=patch_batch_size, track_grad=False)
            D = torch.squeeze(D, dim=0)
            z = randn_like(x)

            if ddnm:
                Dsmall = D[:,pad:pad+w, pad:pad+w]
                x0hat = inverseop.Adagger(noisy) + Dsmall - inverseop.Adagger(inverseop.A(Dsmall))
                x0hat = torch.nn.functional.pad(x0hat, (pad, pad, pad, pad), "constant", 0)
                score = (x0hat-x)/t_cur**2
            else:
                score = (D-x)/t_cur**2
                Ainput = noisy-inverseop.A(x[:,pad:pad+w, pad:pad+w].to(dtype=torch.float32))
                stepsize = zeta/torch.norm(Ainput)* min(40, t_cur*200)
                x[:,pad:pad+w, pad:pad+w] += stepsize * inverseop.AT(Ainput)

            if i < num_steps - 1:
                x = x + alpha/2 * score + torch.sqrt(alpha) * z
            else:
                x = x + alpha/2 * score
        if intermediate_dir and intermediate_interval > 0:
            if i % intermediate_interval == 0 or i == num_steps-1:
                makeFigures(
                    x_init,
                    x[:,pad:pad+w, pad:pad+w],
                    clean,
                    i,
                    preview_dir=intermediate_dir,
                )
    return x


#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value

#----------------------------------------------------------------------------

def fill_lion_metadata_defaults(network_pkl, image_size, pad, psize, channels):
    metadata = {}
    if network_pkl.endswith('.pt') or network_pkl.endswith('.pth'):
        metadata = read_lion_padis_metadata(network_pkl)

    image_size = image_size if image_size is not None else metadata.get('image_size')
    pad = pad if pad is not None else metadata.get('pad')
    psize = psize if psize is not None else metadata.get('psize')
    channels = channels if channels is not None else metadata.get('channels', 1)

    missing = []
    if image_size is None:
        missing.append('--image_size')
    if pad is None:
        missing.append('--pad')
    if psize is None:
        missing.append('--psize')
    if missing:
        raise click.ClickException(
            'Missing required reconstruction size option(s): '
            + ', '.join(missing)
            + '. Pass them explicitly, or use a LION checkpoint with a .json sidecar.'
        )
    return int(image_size), int(pad), int(psize), int(channels), metadata

#----------------------------------------------------------------------------

@click.command()
#directory based options
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--image_dir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--image_size',                help='Sample resolution', metavar='INT',                                 type=int, default=None)
@click.option('--pad',                help='Pad width', metavar='INT',                                 type=int, default=None)
@click.option('--psize',                help='Patch size', metavar='INT',                                 type=int, default=None)
@click.option('--device',                help='Torch device', metavar='STR',                                 type=str, default='cuda', show_default=True)
@click.option('--lion_repo',             help='Local LION repository path for .pt checkpoints', metavar='DIR', type=str, default=None)
@click.option('--raw_weights',           help='Ignore EMA sidecar/state when loading LION checkpoints', is_flag=True)
@click.option('--max_images',            help='Maximum number of PNG images to reconstruct', type=int, default=None)
@click.option('--start_index',           help='Index in sorted PNG list to start from', type=int, default=0, show_default=True)
@click.option('--seed',                  help='Seed Python, NumPy, and PyTorch RNGs for reproducible patch/noise draws. Omit to keep original unseeded Python-random behavior.', type=int, default=None)
@click.option('--ct_impl',               help='ODL RayTransform implementation. Defaults to astra_cuda on CUDA and astra_cpu on CPU.', type=click.Choice(['astra_cuda', 'astra_cpu', 'skimage']), default=None)
@click.option('--intermediate_dir',       help='Directory for intermediate sampler PNGs. Defaults to OUTDIR/intermediates. Use --intermediate_interval 0 to disable.', type=str, default=None)
@click.option('--intermediate_interval',  help='Save intermediate PNGs every N outer steps. Set 0 to disable.', type=click.IntRange(min=0), default=5, show_default=True)
@click.option('--trace_file',             help='Optional JSON file for public sampler tensor statistics.', type=str, default=None)
@click.option('--trace_interval',         help='Write public sampler tensor statistics every N outer steps. Set 0 to disable.', type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--stop_after_outer_steps', help='Debugging aid: stop after this many outer sampler steps while preserving the full sigma schedule.', type=click.IntRange(min=1), default=None)
@click.option('--data_gradient_scale',    help='Optional multiplier for the DPS norm-gradient. Defaults to the inverse operator compatibility scale.', type=float, default=None)
@click.option('--patch_batch_size',       help='Optional patch denoiser microbatch size for public helper samplers.', type=click.IntRange(min=1), default=None)
@click.option('--patch_overlap',          help='Overlap in pixels for patch_average/patch_stitch helper samplers.', type=click.IntRange(min=0), default=8, show_default=True)
@click.option('--checkpoint_denoiser',    help='Use activation checkpointing for patch_average/patch_stitch denoiser microbatches.', is_flag=True)

#inverse operator options
@click.option('--views',                help='Number of CT views', metavar='INT',                                type=click.IntRange(min=1), default=20, show_default=True)
@click.option('--blursize',                help='Size of blur kernel', metavar='INT',                                type=click.IntRange(min=1), default=31, show_default=True)
@click.option('--channels',                help='Image channels', metavar='INT',                                type=click.IntRange(min=1), default=None)
@click.option('--name',                  help='Experiment type', metavar='ct_parbeam|ct_fanbeam|ct_lion_fanbeam|ct_lion_parbeam|denoise',             type=click.Choice(['ct_parbeam', 'ct_fanbeam', 'ct_lion_fanbeam', 'ct_lion_parbeam', 'lact', 'denoise', 'deblur_uniform', 'super']))
@click.option('--sigma',                help='Noise of measurement', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--scale',                help='Superresolution scale', metavar='INT',                                type=click.IntRange(min=1), default=2, show_default=True)

@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)

#solver options
@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)
@click.option('--zeta',                help='Step size', metavar='FLOAT',                          type=click.FloatRange(min=0), default=1.0, show_default=True)
@click.option('--sampler',             help='Public helper sampler to execute.', type=click.Choice(['dps', 'pc', 'langevin', 'ddnm', 'patch_average', 'patch_stitch']), default='dps', show_default=True)

@click.option('--solver',                  help='Ablate ODE solver', metavar='euler|heun',                          type=click.Choice(['euler', 'heun']))
@click.option('--disc', 'discretization',  help='Ablate time step discretization {t_i}', metavar='vp|ve|iddpm|edm', type=click.Choice(['vp', 've', 'iddpm', 'edm']))
@click.option('--schedule',                help='Ablate noise schedule sigma(t)', metavar='vp|ve|linear',           type=click.Choice(['vp', 've', 'linear']))
@click.option('--scaling',                 help='Ablate signal scaling s(t)', metavar='vp|none',                    type=click.Choice(['vp', 'none']))

def main(network_pkl, image_size, outdir, image_dir, name, views, blursize, scale, channels, sigma, pad, psize,
         device, lion_repo, raw_weights, max_images, start_index, seed, ct_impl,
         intermediate_dir, intermediate_interval, trace_file, trace_interval, stop_after_outer_steps, data_gradient_scale, patch_batch_size, patch_overlap, checkpoint_denoiser, sampler, **sampler_kwargs):
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available.')
    if ct_impl is None:
        ct_impl = 'astra_cuda' if device.type == 'cuda' else 'astra_cpu'
    os.makedirs(outdir, exist_ok=True)
    set_run_seed(seed)
    if seed is not None:
        log(f'Using seed={seed}')
    log(f'Using torch device={device}, ODL CT implementation={ct_impl}')
    if intermediate_interval > 0:
        if intermediate_dir is None:
            intermediate_dir = os.path.join(outdir, 'intermediates')
        os.makedirs(intermediate_dir, exist_ok=True)
        log(
            f'Saving intermediate PNGs every {intermediate_interval} outer steps '
            f'to "{intermediate_dir}".'
        )
    else:
        intermediate_dir = None
        log('Intermediate PNG saving disabled.')
    if trace_interval > 0:
        if trace_file is None:
            trace_file = os.path.join(outdir, 'trace.json')
        log(f'Saving public sampler trace every {trace_interval} outer steps to "{trace_file}".')
    else:
        trace_file = None

    image_size, pad, psize, channels, lion_metadata = fill_lion_metadata_defaults(
        network_pkl, image_size, pad, psize, channels
    )
    if lion_metadata:
        log(
            'Using LION checkpoint metadata: '
            f'image_size={image_size}, pad={pad}, psize={psize}, channels={channels}'
        )

    # Load network.
    log(f'Loading network from "{network_pkl}"...')
    if network_pkl.endswith('.pt') or network_pkl.endswith('.pth'):
        net = load_lion_padis_model(
            network_pkl,
            device=device,
            lion_repo=lion_repo,
            use_ema=not raw_weights,
        )
    else:
        with dnnlib.util.open_url(network_pkl, verbose=False) as f:
            net = pickle.load(f)['ema'].to(device)
    log('Network loaded.')

    files = os.listdir(image_dir)
    png_files = sorted(file for file in files if file.lower().endswith('.png'))
    if start_index:
        png_files = png_files[start_index:]
    if max_images is not None:
        png_files = png_files[:max_images]
    if not png_files:
        raise click.ClickException(f'No PNG files found in {image_dir}')
    log(f'Found {len(png_files)} PNG image(s) in {image_dir}.')
    log(f'Building inverse operator for name={name}, views={views}, image_size={image_size}.')
    inverseop = InverseOperator(image_size, name, views=views, channels=channels, blursize=blursize, scale_factor=scale, ct_impl=ct_impl)
    log('Inverse operator ready.')
    effective_data_gradient_scale = (
        getattr(inverseop, 'data_gradient_scale', 1.0)
        if data_gradient_scale is None
        else data_gradient_scale
    )
    if effective_data_gradient_scale != 1.0:
        log(f'Using DPS data gradient scale={effective_data_gradient_scale}.')

    x_start = 0
    y_start = 0
    resolution = image_size + 2*pad
    x_pos = torch.arange(x_start, x_start+resolution).view(1, -1).repeat(resolution, 1)
    y_pos = torch.arange(y_start, y_start+resolution).view(-1, 1).repeat(1, resolution)
    x_pos = (x_pos / (resolution - 1) - 0.5) * 2.
    y_pos = (y_pos / (resolution - 1) - 0.5) * 2.
    latents_pos = torch.stack([x_pos, y_pos], dim=0).to(device)
    latents_pos = latents_pos.unsqueeze(0).repeat(1, 1, 1, 1)

    allclean = np.zeros((len(png_files), image_size, image_size, channels))
    allrecon = np.zeros((len(png_files), image_size, image_size, channels))
    log(f'Generating images to "{outdir}"...')
    totpsnr = 0
    totssim = 0
    psnrarr = []
    ssimarr = []

    for loop in tqdm.tqdm(
        range(len(png_files)),
        total=len(png_files),
        desc='PaDIS input images',
        dynamic_ncols=True,
        file=sys.stdout,
    ):
        clean = PIL.Image.open(os.path.join(image_dir, png_files[loop]))
        clean = np.asarray(clean)/255
        if channels == 1:
            clean = np.expand_dims(clean, 0)
        elif channels == 3:
            clean = np.transpose(clean, (2,0,1))
        log(f'Input range: min={clean.min()} max={clean.max()}')
        log(f'clean shape: {clean.shape}')

        log(f'Now doing image "{png_files[loop]}"')

        xclean = torch.from_numpy(clean).to(device=device, dtype=torch.float32)
        log('Forward-projecting clean image.')
        noisy_y = inverseop.A(xclean)
        log(f'clean tensor shape: {tuple(xclean.shape)}')
        log(f'noisy measurement shape: {tuple(noisy_y.shape)}')
        noisy_y = noisy_y + sigma*torch.randn_like(noisy_y)
        scipy.io.savemat('proj.mat', {'proj': noisy_y.cpu().numpy()})

        latents = torch.randn([1, channels, image_size, image_size], device=device)

        sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
        log(f'Starting {sampler} sampler.')
        sample_intermediate_dir = None
        if intermediate_dir is not None:
            sample_intermediate_dir = os.path.join(
                intermediate_dir,
                os.path.splitext(png_files[loop])[0],
            )
        if sampler == 'dps':
            images = dps(
                net,
                latents,
                latents_pos,
                inverseop,
                clean=clean,
                noisy=noisy_y,
                pad=pad,
                psize=psize,
                intermediate_dir=sample_intermediate_dir,
                intermediate_interval=intermediate_interval,
                data_gradient_scale=effective_data_gradient_scale,
                trace_file=trace_file,
                trace_interval=trace_interval,
                stop_after_outer_steps=stop_after_outer_steps,
                patch_batch_size=patch_batch_size,
                **sampler_kwargs,
            )
        elif sampler == 'pc':
            images = pc_sampling(
                net,
                latents,
                latents_pos,
                inverseop,
                clean=clean,
                noisy=noisy_y,
                pad=pad,
                psize=psize,
                intermediate_dir=sample_intermediate_dir,
                intermediate_interval=intermediate_interval,
                stop_after_outer_steps=stop_after_outer_steps,
                patch_batch_size=patch_batch_size,
                trace_file=trace_file,
                trace_interval=trace_interval,
                **sampler_kwargs,
            )
        elif sampler in ('langevin', 'ddnm'):
            images = langevin(
                net,
                latents,
                latents_pos,
                inverseop,
                clean=clean,
                noisy=noisy_y,
                pad=pad,
                psize=psize,
                ddnm=(sampler == 'ddnm'),
                intermediate_dir=sample_intermediate_dir,
                intermediate_interval=intermediate_interval,
                stop_after_outer_steps=stop_after_outer_steps,
                patch_batch_size=patch_batch_size,
                **sampler_kwargs,
            )
        elif sampler in ('patch_average', 'patch_stitch'):
            images = fixed_patch_dps(
                net,
                latents,
                latents_pos,
                inverseop,
                clean=clean,
                noisy=noisy_y,
                pad=pad,
                psize=psize,
                intermediate_dir=sample_intermediate_dir,
                intermediate_interval=intermediate_interval,
                data_gradient_scale=effective_data_gradient_scale,
                trace_file=trace_file,
                trace_interval=trace_interval,
                stop_after_outer_steps=stop_after_outer_steps,
                patch_batch_size=patch_batch_size,
                sampler=sampler,
                overlap=patch_overlap,
                checkpoint_denoiser=checkpoint_denoiser,
                **sampler_kwargs,
            )
        else:
            raise click.ClickException(f'Unknown sampler: {sampler}')
        log(f'{sampler} sampler finished.')

        images = torch.clamp(images, min=0, max=1)
        images = images[:, pad:pad+image_size, pad:pad+image_size]
        images = torch.permute(images, (1,2,0))
        images = images.cpu().numpy()
        cleantmp = np.transpose(clean, (1,2,0))
        thispsnr = psnr(images, cleantmp, data_range=1)
        log(f'psnr for this image: {thispsnr}')
        myssim = ssim(images, cleantmp, channel_axis=2, data_range=1)
        log(f'ssim for this image: {myssim}')
        totpsnr += thispsnr
        totssim += myssim
        psnrarr.append(thispsnr)
        ssimarr.append(myssim)

        allclean[loop, :,:,:] = cleantmp
        allrecon[loop,:,:,:] = images
        PIL.Image.fromarray(np.uint8(np.round(np.squeeze(images) * 255.0))).save(
            os.path.join(outdir, f'{os.path.splitext(png_files[loop])[0]}_recon.png')
        )

    log(f'average psnr: {totpsnr/(len(png_files))}')
    log(f'average ssim: {totssim/(len(png_files))}')
    np.savez_compressed(
        os.path.join(outdir, 'reconstructions.npz'),
        clean=allclean,
        recon=allrecon,
        psnr=np.asarray(psnrarr),
        ssim=np.asarray(ssimarr),
        files=np.asarray(png_files),
    )


#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
