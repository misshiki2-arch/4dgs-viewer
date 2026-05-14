#!/usr/bin/env python3
"""
compare_png.py

Compare two PNG images and output numeric difference metrics.

Typical use:
  python tools/compare_png.py \
    --a /home/demo/work/json/step107_000151_v13_canvas.png \
    --b /home/demo/work/json/step105_000151_v13_canvas.png \
    --out /home/demo/work/json/step107_vs_step105_absdiff.png \
    --json /home/demo/work/json/step107_vs_step105_image_diff_summary.json

Notes:
- Images are converted to RGBA before comparison.
- Difference metrics are computed over all RGBA channels.
- nonBlackPixelCount is computed from RGB only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rgba(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with Image.open(path) as img:
        rgba = img.convert("RGBA")
        return np.asarray(rgba, dtype=np.uint8)


def count_nonblack_pixels(arr: np.ndarray, threshold: int = 0) -> int:
    """
    Count pixels whose RGB channels contain at least one value above threshold.
    Alpha is ignored for this count.
    """
    rgb = arr[..., :3]
    mask = np.any(rgb > threshold, axis=2)
    return int(np.count_nonzero(mask))


def build_absdiff_image(diff: np.ndarray, scale: float) -> Image.Image:
    """
    Build an RGBA absolute-difference image.
    RGB channels show absolute RGB difference.
    Alpha is set to 255.
    """
    rgb = diff[..., :3].astype(np.float32) * scale
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    alpha = np.full((rgb.shape[0], rgb.shape[1], 1), 255, dtype=np.uint8)
    rgba = np.concatenate([rgb, alpha], axis=2)
    return Image.fromarray(rgba, mode="RGBA")


def compare_images(
    a_path: Path,
    b_path: Path,
    nonblack_threshold: int,
    diff_scale: float,
    out_path: Path | None,
) -> Dict[str, Any]:
    a = load_rgba(a_path)
    b = load_rgba(b_path)

    summary: Dict[str, Any] = {
        "a": str(a_path),
        "b": str(b_path),
        "aSha256": sha256_file(a_path),
        "bSha256": sha256_file(b_path),
        "sameSha256": sha256_file(a_path) == sha256_file(b_path),
        "aShape": list(a.shape),
        "bShape": list(b.shape),
        "sameSize": list(a.shape) == list(b.shape),
        "nonBlackThreshold": nonblack_threshold,
    }

    summary["aNonBlackPixelCount"] = count_nonblack_pixels(a, nonblack_threshold)
    summary["bNonBlackPixelCount"] = count_nonblack_pixels(b, nonblack_threshold)

    if a.shape != b.shape:
        summary.update(
            {
                "comparable": False,
                "error": "Image sizes differ. Pixel-wise metrics were not computed.",
            }
        )
        return summary

    a_i = a.astype(np.int16)
    b_i = b.astype(np.int16)
    diff = np.abs(a_i - b_i).astype(np.uint8)

    diff_float = diff.astype(np.float64)

    different_any_channel = np.any(diff > 0, axis=2)
    different_all_channels = np.all(diff > 0, axis=2)

    summary.update(
        {
            "comparable": True,
            "width": int(a.shape[1]),
            "height": int(a.shape[0]),
            "channelCount": int(a.shape[2]),
            "pixelCount": int(a.shape[0] * a.shape[1]),
            "valueCount": int(a.size),
            "mae": float(np.mean(diff_float)),
            "rmse": float(math.sqrt(np.mean(diff_float ** 2))),
            "maxAbsError": int(np.max(diff)),
            "sumAbsError": int(np.sum(diff.astype(np.uint64))),
            "differentPixelCountAnyChannel": int(np.count_nonzero(different_any_channel)),
            "differentPixelCountAllChannels": int(np.count_nonzero(different_all_channels)),
            "differentValueCount": int(np.count_nonzero(diff)),
            "samePixels": int(a.shape[0] * a.shape[1] - np.count_nonzero(different_any_channel)),
        }
    )

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        diff_img = build_absdiff_image(diff, diff_scale)
        diff_img.save(out_path)
        summary["absDiffImage"] = str(out_path)
        summary["absDiffScale"] = diff_scale

    return summary


def print_human_summary(summary: Dict[str, Any]) -> None:
    print("PNG comparison summary")
    print(f"- A: {summary['a']}")
    print(f"- B: {summary['b']}")
    print(f"- sameSha256: {summary['sameSha256']}")
    print(f"- sameSize: {summary['sameSize']}")
    print(f"- A nonBlackPixelCount: {summary['aNonBlackPixelCount']}")
    print(f"- B nonBlackPixelCount: {summary['bNonBlackPixelCount']}")

    if not summary.get("comparable", False):
        print(f"- comparable: false")
        print(f"- error: {summary.get('error')}")
        return

    print(f"- size: {summary['width']}x{summary['height']}")
    print(f"- MAE: {summary['mae']}")
    print(f"- RMSE: {summary['rmse']}")
    print(f"- maxAbsError: {summary['maxAbsError']}")
    print(f"- differentPixelCountAnyChannel: {summary['differentPixelCountAnyChannel']}")
    print(f"- differentValueCount: {summary['differentValueCount']}")
    if "absDiffImage" in summary:
        print(f"- absDiffImage: {summary['absDiffImage']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two PNG images and output difference metrics."
    )
    parser.add_argument("--a", required=True, help="First PNG path.")
    parser.add_argument("--b", required=True, help="Second PNG path.")
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output path for absolute-difference PNG.",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="Optional output path for JSON summary.",
    )
    parser.add_argument(
        "--diff-scale",
        type=float,
        default=1.0,
        help="Scale factor for saved absdiff image. Default: 1.0",
    )
    parser.add_argument(
        "--nonblack-threshold",
        type=int,
        default=0,
        help="RGB threshold for non-black pixel count. Default: 0",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    a_path = Path(args.a)
    b_path = Path(args.b)
    out_path = Path(args.out) if args.out else None
    json_path = Path(args.json) if args.json else None

    summary = compare_images(
        a_path=a_path,
        b_path=b_path,
        nonblack_threshold=args.nonblack_threshold,
        diff_scale=args.diff_scale,
        out_path=out_path,
    )

    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print_human_summary(summary)

    return 0 if summary.get("comparable", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
