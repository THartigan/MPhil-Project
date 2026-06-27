"""Prepare LION-preprocessed LIDC-IDRI slices as PNGs for original PaDIS."""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def hu_to_lion_normal(image: np.ndarray) -> np.ndarray:
    return np.clip((image.astype(np.float32) + 1000.0) / 3000.0, 0.0, 1.0)


def insert_lion_repo(lion_repo: pathlib.Path | None) -> pathlib.Path:
    candidates = []
    if lion_repo is not None:
        candidates.append(lion_repo.expanduser())
    candidates.append(pathlib.Path(__file__).resolve().parents[1] / "LION")

    for candidate in candidates:
        if (candidate / "LION").is_dir():
            path = str(candidate.resolve())
            if path not in sys.path:
                sys.path.insert(0, path)
            return candidate.resolve()

    tried = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find local LION repo. Tried:\n  {tried}")


def save_normal_png(image: np.ndarray | torch.Tensor, output_path: pathlib.Path) -> None:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().squeeze().numpy()
    image = np.clip(image.astype(np.float32), 0.0, 1.0)
    pil_image = Image.fromarray(np.uint8(np.round(image * 255.0)), mode="L")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_image.save(output_path)


def resize_hu_like_lion(image: np.ndarray, image_size: int) -> np.ndarray:
    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
    if tuple(tensor.shape[-2:]) != (image_size, image_size):
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
    return tensor.squeeze(0).squeeze(0).numpy()


def convert_raw_slice(path: pathlib.Path, output_path: pathlib.Path, image_size: int) -> None:
    hu_image = resize_hu_like_lion(np.load(path), image_size)
    save_normal_png(hu_to_lion_normal(hu_image), output_path)


def export_with_lion_loader(args) -> None:
    insert_lion_repo(args.lion_repo)
    from LION.CTtools.ct_geometry import Geometry
    from LION.data_loaders.LIDC_IDRI import LIDC_IDRI

    geometry = Geometry.default_parameters(image_scaling=args.image_size / 512.0)
    params = LIDC_IDRI.default_parameters(geometry=geometry, task="image_prior")
    params.device = torch.device("cpu")
    if args.input_root is not None:
        params.folder = args.input_root

    splits = ["train", "validation", "test"] if args.split == "all" else [args.split]
    written = 0
    for split in splits:
        dataset = LIDC_IDRI(split, parameters=params, geometry_parameters=geometry)
        stop = len(dataset) if args.limit is None else min(len(dataset), args.start + args.limit)
        for index in tqdm(range(args.start, stop), desc=f"LION {split} PNGs"):
            patient_id = dataset.slice_index_to_patient_id_list[index]
            first_index = dataset.patient_id_to_first_index_dict[patient_id]
            slice_index = dataset.slices_to_load[patient_id][index - first_index]
            _, image = dataset[index]
            output_name = f"{split}_{patient_id}_slice_{int(slice_index)}.png"
            save_normal_png(image, args.output_dir / output_name)
            written += 1
    print(f"Wrote {written} LION-preprocessed PNG files to {args.output_dir}")


def export_raw_glob(args) -> None:
    files = sorted(args.input_root.expanduser().glob("*/slice_*.npy"))
    if not files:
        raise FileNotFoundError(f"No slice_*.npy files found under {args.input_root}")
    stop = None if args.limit is None else args.start + args.limit
    selected = files[args.start : stop]
    for path in tqdm(selected, desc="Raw LIDC PNGs"):
        output_name = f"{path.parent.name}_{path.stem}.png"
        convert_raw_slice(path, args.output_dir / output_name, args.image_size)
    print(f"Wrote {len(selected)} PNG files to {args.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("lion-loader", "raw-glob"),
        default="lion-loader",
        help="Use LION's LIDC_IDRI dataset path by default, including split and slice selection.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test", "all"),
        default="test",
        help="LION dataset split to export when --source=lion-loader.",
    )
    parser.add_argument(
        "--input-root",
        type=pathlib.Path,
        default=PROJECT_ROOT / "Data/processed/LIDC-IDRI",
        help="Root containing LIDC-IDRI patient folders with slice_*.npy files.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=PROJECT_ROOT / "Data/processed/LIDC-IDRI-padis-png-256",
        help="Directory to write PNG files into.",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--lion-repo", type=pathlib.Path, default=None)
    args = parser.parse_args()

    if args.source == "lion-loader":
        export_with_lion_loader(args)
    else:
        export_raw_glob(args)


if __name__ == "__main__":
    main()
