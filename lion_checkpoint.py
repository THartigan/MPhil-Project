"""Adapter for running LION PaDIS checkpoints in the original PaDIS sampler."""

from __future__ import annotations

import json
import pathlib
import sys
import warnings

import torch


def _insert_lion_repo(lion_repo: str | pathlib.Path | None) -> pathlib.Path:
    candidates = []
    if lion_repo is not None:
        candidates.append(pathlib.Path(lion_repo).expanduser())
    candidates.append(pathlib.Path(__file__).resolve().parents[1] / "LION")

    for candidate in candidates:
        if (candidate / "LION").is_dir():
            path = str(candidate.resolve())
            if path not in sys.path:
                sys.path.insert(0, path)
            return candidate.resolve()

    tried = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find local LION repo. Tried:\n  {tried}")


def _torch_load(path: pathlib.Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def read_lion_padis_metadata(checkpoint_path: str | pathlib.Path) -> dict:
    """Read reconstruction-relevant metadata from a LION PaDIS sidecar JSON."""

    checkpoint_path = pathlib.Path(checkpoint_path).expanduser()
    json_path = checkpoint_path.with_suffix(".json")
    if not json_path.is_file():
        return {}

    with open(json_path, encoding="utf-8") as f:
        options = json.load(f)

    model_params = options.get("model_parameters", {})
    geometry = options.get("geometry", {})
    image_shape = geometry.get("image_shape")
    metadata = {}
    if isinstance(image_shape, list) and len(image_shape) >= 3:
        metadata["channels"] = int(image_shape[0])
        metadata["image_size"] = int(image_shape[-1])
    if "pad_width" in model_params:
        metadata["pad"] = int(model_params["pad_width"])
    if "largest_patch_size" in model_params:
        metadata["psize"] = int(model_params["largest_patch_size"])
    return metadata


class LionEDMDenoiserAdapter(torch.nn.Module):
    """Expose a raw LION NCSN++ checkpoint as original PaDIS EDM denoiser."""

    def __init__(self, model: torch.nn.Module, sigma_data: float = 0.5):
        super().__init__()
        self.model = model
        self.sigma_data = float(sigma_data)
        params = getattr(model, "model_parameters", None)
        self.sigma_min = float(getattr(params, "sigma_min", 0.0))
        self.sigma_max = float(getattr(params, "sigma_max", float("inf")))

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        x_pos: torch.Tensor | None = None,
        class_labels=None,
        force_fp32: bool = False,
        **model_kwargs,
    ) -> torch.Tensor:
        del class_labels, force_fp32, model_kwargs
        x = x.to(torch.float32)
        sigma_vec = torch.as_tensor(sigma, device=x.device, dtype=x.dtype).flatten()
        if sigma_vec.numel() == 1:
            sigma_vec = sigma_vec.expand(x.shape[0])
        if sigma_vec.numel() != x.shape[0]:
            raise ValueError(
                f"sigma has {sigma_vec.numel()} values for batch size {x.shape[0]}"
            )

        sigma_view = sigma_vec.reshape(x.shape[0], 1, 1, 1)
        sigma_data = torch.as_tensor(self.sigma_data, device=x.device, dtype=x.dtype)
        c_skip = sigma_data.square() / (sigma_view.square() + sigma_data.square())
        c_out = sigma_view * sigma_data / (
            sigma_view.square() + sigma_data.square()
        ).sqrt()
        c_in = 1 / (sigma_data.square() + sigma_view.square()).sqrt()
        c_noise = sigma_vec.log() / 4

        if x_pos is not None:
            model_input = torch.cat((c_in * x, x_pos.to(device=x.device, dtype=x.dtype)), dim=1)
        else:
            model_input = c_in * x
        model_output = self.model(model_input, c_noise)
        return c_skip * x + c_out * model_output.to(torch.float32)

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


def load_lion_padis_model(
    checkpoint_path: str | pathlib.Path,
    *,
    device: torch.device,
    lion_repo: str | pathlib.Path | None = None,
    use_ema: bool = True,
) -> LionEDMDenoiserAdapter:
    """Load a LION PaDIS `.pt` checkpoint and wrap it for original PaDIS code."""

    checkpoint_path = pathlib.Path(checkpoint_path).expanduser().resolve()
    _insert_lion_repo(lion_repo)

    from LION.CTtools.ct_geometry import Geometry
    from LION.models.diffusion import NCSNpp
    from LION.utils.parameter import LIONParameter

    json_path = checkpoint_path.with_suffix(".json")
    if json_path.is_file():
        options = LIONParameter()
        options.load(json_path)
        model_params = options.model_parameters
        geometry = Geometry.init_from_parameter(options.geometry)
    else:
        warnings.warn(
            f"No sidecar JSON found at {json_path}; using PaDIS LIDC 256 defaults.",
            stacklevel=2,
        )
        model_params = NCSNpp.default_parameters("padis-paper-ct-256")
        geometry = Geometry.default_parameters(image_scaling=0.5)

    model = NCSNpp(model_params, geometry).to(device)
    checkpoint = _torch_load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    ema_state = checkpoint.get("ema_state_dict") if isinstance(checkpoint, dict) else None
    ema_path = checkpoint_path.with_suffix(".ema.pt")
    if use_ema and ema_state is None and ema_path.is_file():
        ema_checkpoint = _torch_load(ema_path, map_location=device)
        ema_state = ema_checkpoint.get("ema_state_dict")
    if use_ema and ema_state is not None:
        state_dict = dict(state_dict)
        state_dict.update(ema_state)
        print("Loaded EMA weights from LION checkpoint.")

    model.load_state_dict(state_dict)
    model.eval()
    adapter = LionEDMDenoiserAdapter(model).to(device).eval()
    adapter.lion_metadata = read_lion_padis_metadata(checkpoint_path)
    return adapter
