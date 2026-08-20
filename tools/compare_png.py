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
- Difference metrics use the selected comparison channel mode (RGB or RGBA).
- The default comparison channel mode remains RGBA.
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


COMPARISON_CHANNEL_COUNTS = {
    "rgb": 3,
    "rgba": 4,
}


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


def image_stats(arr: np.ndarray, threshold: int = 0) -> Dict[str, Any]:
    rgb = arr[..., :3]
    alpha = arr[..., 3]
    return {
        "width": int(arr.shape[1]),
        "height": int(arr.shape[0]),
        "channelCount": int(arr.shape[2]),
        "pixelCount": int(arr.shape[0] * arr.shape[1]),
        "nonBlackPixelCount": count_nonblack_pixels(arr, threshold),
        "nonBlackPixelRatio": float(
            count_nonblack_pixels(arr, threshold) / max(1, arr.shape[0] * arr.shape[1])
        ),
        "rgbMin": int(np.min(rgb)),
        "rgbMax": int(np.max(rgb)),
        "rgbMean": float(np.mean(rgb)),
        "alphaMin": int(np.min(alpha)),
        "alphaMax": int(np.max(alpha)),
        "alphaMean": float(np.mean(alpha)),
    }


def load_json_object(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def classify_difference(
    *,
    same_size: bool,
    conditions_ready: bool,
    condition_mismatch_reason: str | None,
    a_nonblack: int,
    b_nonblack: int,
    differing_ratio: float | None,
) -> str:
    if not same_size:
        return "viewport-or-resolution-mismatch"
    if not conditions_ready:
        return condition_mismatch_reason or "comparison-condition-mismatch"
    if a_nonblack > 0 and b_nonblack == 0:
        return "runtime-output-nonblank-but-saved-png-blank"
    if a_nonblack == 0 and b_nonblack > 0:
        return "reference-image-source-mismatch"
    if differing_ratio is None:
        return "comparison-condition-mismatch"
    if differing_ratio == 0:
        return "parity-candidate"
    return "comparison-ready-difference-unclassified"


def classify_reference_source_kind(path_or_source: str | None) -> str:
    if not path_or_source:
        return "unknown"
    normalized = path_or_source.replace("\\", "/")
    if "/outputs/" in normalized and "cuda_reference" in normalized:
        if normalized.endswith("_render.png"):
            return "cuda-reference-render-output"
        return "cuda-reference-derived-output"
    if "/data/4dgs_sph_scene/images/" in normalized:
        return "dataset-fixed-reference-image"
    if normalized.endswith("_gt.png"):
        return "dataset-ground-truth-image"
    return "explicit-reference-image"


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


def comparison_channel_count(comparison_channel_mode: str) -> int:
    try:
        return COMPARISON_CHANNEL_COUNTS[comparison_channel_mode]
    except KeyError as error:
        supported = ", ".join(sorted(COMPARISON_CHANNEL_COUNTS))
        raise ValueError(
            f"unsupported comparison channel mode {comparison_channel_mode!r}; "
            f"expected one of: {supported}"
        ) from error


def compute_pixel_metrics(
    a: np.ndarray,
    b: np.ndarray,
    comparison_channel_mode: str = "rgba",
) -> Dict[str, Any]:
    channel_count = comparison_channel_count(comparison_channel_mode)
    if a.shape != b.shape:
        raise ValueError(
            f"comparison arrays must have matching shapes: {a.shape} != {b.shape}"
        )
    if a.ndim != 3 or a.shape[2] < channel_count:
        raise ValueError(
            "comparison arrays must provide the requested channel count: "
            f"shape={a.shape}, requested={channel_count}"
        )
    a_values = a[..., :channel_count]
    b_values = b[..., :channel_count]
    diff = np.abs(a_values.astype(np.int16) - b_values.astype(np.int16)).astype(
        np.uint8
    )
    diff_float = diff.astype(np.float64)
    different_any_channel = np.any(diff > 0, axis=2)
    different_all_channels = np.all(diff > 0, axis=2)
    return {
        "comparisonChannelCount": channel_count,
        "valueCount": int(a_values.size),
        "mae": float(np.mean(diff_float)),
        "rmse": float(math.sqrt(np.mean(diff_float ** 2))),
        "maxAbsError": int(np.max(diff)),
        "sumAbsError": int(np.sum(diff.astype(np.uint64))),
        "differentPixelCountAnyChannel": int(np.count_nonzero(different_any_channel)),
        "differentPixelCountAllChannels": int(np.count_nonzero(different_all_channels)),
        "differentValueCount": int(np.count_nonzero(diff)),
        "samePixels": int(a.shape[0] * a.shape[1] - np.count_nonzero(different_any_channel)),
        "differentPixelRatioAnyChannel": float(
            np.count_nonzero(different_any_channel) / max(1, a.shape[0] * a.shape[1])
        ),
    }


def compare_images(
    a_path: Path,
    b_path: Path,
    nonblack_threshold: int,
    diff_scale: float,
    out_path: Path | None,
    *,
    reference_source: str | None,
    webgpu_source: str | None,
    camera_label: str | None,
    frame_label: str | None,
    dataset_time: str | None,
    viewport: str | None,
    background_policy: str | None,
    color_space_policy: str | None,
    pixel_origin: str,
    y_coordinate_convention: str,
    screen_space_convention: str | None,
    capture_source: str | None,
    fixed_reference_camera_mode: str | None,
    webgpu_camera_constants_source: str | None,
    comparison_channel_mode: str,
    reference_source_kind: str | None,
    webgpu_source_kind: str | None,
    conditions_json: Path | None,
    include_vertical_flip_diagnostic: bool,
) -> Dict[str, Any]:
    comparison_channel_count(comparison_channel_mode)
    a = load_rgba(a_path)
    b = load_rgba(b_path)
    conditions_from_json = load_json_object(conditions_json)

    summary: Dict[str, Any] = {
        "schemaVersion": "phase3-step110-fixed-condition-png-comparison-v1",
        "a": str(a_path),
        "b": str(b_path),
        "referenceImagePath": str(a_path),
        "webgpuPngPath": str(b_path),
        "referenceImageSource": reference_source or str(a_path),
        "referenceImageSourceKind": reference_source_kind
        or classify_reference_source_kind(reference_source or str(a_path)),
        "webgpuPngSource": webgpu_source or str(b_path),
        "webgpuPngSourceKind": webgpu_source_kind or "webgpu-saved-png",
        "aSha256": sha256_file(a_path),
        "bSha256": sha256_file(b_path),
        "sameSha256": sha256_file(a_path) == sha256_file(b_path),
        "aShape": list(a.shape),
        "bShape": list(b.shape),
        "sameSize": list(a.shape) == list(b.shape),
        "nonBlackThreshold": nonblack_threshold,
        "comparisonChannelMode": comparison_channel_mode,
        "comparisonConditions": {
            "cameraLabel": camera_label,
            "frameLabel": frame_label,
            "datasetTime": dataset_time,
            "viewport": viewport,
            "backgroundPolicy": background_policy,
            "colorSpacePolicy": color_space_policy,
            "pixelOrigin": pixel_origin,
            "yCoordinateConvention": y_coordinate_convention,
            "screenSpaceConvention": screen_space_convention,
            "captureSource": capture_source,
            "fixedReferenceCameraMode": fixed_reference_camera_mode,
            "webgpuCameraConstantsSource": webgpu_camera_constants_source,
            **conditions_from_json,
        },
    }

    summary["aNonBlackPixelCount"] = count_nonblack_pixels(a, nonblack_threshold)
    summary["bNonBlackPixelCount"] = count_nonblack_pixels(b, nonblack_threshold)
    summary["referenceImageStats"] = image_stats(a, nonblack_threshold)
    summary["webgpuImageStats"] = image_stats(b, nonblack_threshold)
    conditions_ready = all(
        summary["comparisonConditions"].get(key)
        for key in [
            "cameraLabel",
            "frameLabel",
            "backgroundPolicy",
            "colorSpacePolicy",
            "pixelOrigin",
            "yCoordinateConvention",
            "screenSpaceConvention",
            "captureSource",
            "fixedReferenceCameraMode",
            "webgpuCameraConstantsSource",
        ]
    )
    condition_mismatch_reason = None
    if not conditions_ready:
        condition_mismatch_reason = "comparison-condition-mismatch"
    summary["comparisonConditionReady"] = conditions_ready

    if a.shape != b.shape:
        summary.update(
            {
                "comparable": False,
                "error": "Image sizes differ. Pixel-wise metrics were not computed.",
                "visualMismatchClassification": classify_difference(
                    same_size=False,
                    conditions_ready=conditions_ready,
                    condition_mismatch_reason=condition_mismatch_reason,
                    a_nonblack=summary["aNonBlackPixelCount"],
                    b_nonblack=summary["bNonBlackPixelCount"],
                    differing_ratio=None,
                ),
            }
        )
        return summary

    diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.uint8)
    summary.update(
        {
            "comparable": True,
            "width": int(a.shape[1]),
            "height": int(a.shape[0]),
            "channelCount": int(a.shape[2]),
            "loadedChannelCount": int(a.shape[2]),
            "pixelCount": int(a.shape[0] * a.shape[1]),
            "loadedValueCount": int(a.size),
            **compute_pixel_metrics(a, b, comparison_channel_mode),
        }
    )
    if include_vertical_flip_diagnostic:
        flipped_b = np.flipud(b)
        flipped_metrics = compute_pixel_metrics(
            a, flipped_b, comparison_channel_mode
        )
        normal_rmse = summary["rmse"]
        flipped_rmse = flipped_metrics["rmse"]
        lower_error_orientation = (
            "vertical-flipped-webgpu-png"
            if flipped_rmse < normal_rmse
            else "normal-webgpu-png"
            if normal_rmse < flipped_rmse
            else "tie"
        )
        summary["orientationDiagnostic"] = {
            "normal": {
                "comparisonChannelMode": comparison_channel_mode,
                "comparisonChannelCount": summary["comparisonChannelCount"],
                "valueCount": summary["valueCount"],
                "mae": summary["mae"],
                "rmse": summary["rmse"],
                "differentPixelRatioAnyChannel": summary[
                    "differentPixelRatioAnyChannel"
                ],
            },
            "verticalFlip": {
                "comparisonChannelMode": comparison_channel_mode,
                "comparisonChannelCount": flipped_metrics[
                    "comparisonChannelCount"
                ],
                "valueCount": flipped_metrics["valueCount"],
                "mae": flipped_metrics["mae"],
                "rmse": flipped_metrics["rmse"],
                "differentPixelRatioAnyChannel": flipped_metrics[
                    "differentPixelRatioAnyChannel"
                ],
            },
            "lowerErrorOrientation": lower_error_orientation,
            "classification": (
                "orientation-diagnostic-only-not-parity-evidence"
                if lower_error_orientation != "tie"
                else "orientation-diagnostic-tie"
            ),
        }
    summary["visualMismatchClassification"] = classify_difference(
        same_size=True,
        conditions_ready=conditions_ready,
        condition_mismatch_reason=condition_mismatch_reason,
        a_nonblack=summary["aNonBlackPixelCount"],
        b_nonblack=summary["bNonBlackPixelCount"],
        differing_ratio=summary["differentPixelRatioAnyChannel"],
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
    print(f"- comparisonChannelMode: {summary.get('comparisonChannelMode')}")
    print(f"- comparisonConditionReady: {summary.get('comparisonConditionReady')}")
    print(f"- visualMismatchClassification: {summary.get('visualMismatchClassification')}")
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
    parser.add_argument("--reference-source", default=None)
    parser.add_argument("--webgpu-source", default=None)
    parser.add_argument("--camera-label", default=None)
    parser.add_argument("--frame-label", default=None)
    parser.add_argument("--dataset-time", default=None)
    parser.add_argument("--viewport", default=None)
    parser.add_argument("--background-policy", default=None)
    parser.add_argument("--color-space-policy", default=None)
    parser.add_argument("--pixel-origin", default="top-left-saved-png")
    parser.add_argument("--y-coordinate-convention", default="screen-y-down")
    parser.add_argument("--screen-space-convention", default=None)
    parser.add_argument("--capture-source", default=None)
    parser.add_argument("--fixed-reference-camera-mode", default=None)
    parser.add_argument("--webgpu-camera-constants-source", default=None)
    parser.add_argument(
        "--comparison-channel-mode",
        choices=sorted(COMPARISON_CHANNEL_COUNTS),
        default="rgba",
    )
    parser.add_argument("--reference-source-kind", default=None)
    parser.add_argument("--webgpu-source-kind", default=None)
    parser.add_argument(
        "--conditions-json",
        default=None,
        help="Optional JSON object with comparison condition metadata.",
    )
    parser.add_argument(
        "--disable-vertical-flip-diagnostic",
        action="store_true",
        help="Do not include diagnostic metrics for vertically flipping image B.",
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
        reference_source=args.reference_source,
        webgpu_source=args.webgpu_source,
        camera_label=args.camera_label,
        frame_label=args.frame_label,
        dataset_time=args.dataset_time,
        viewport=args.viewport,
        background_policy=args.background_policy,
        color_space_policy=args.color_space_policy,
        pixel_origin=args.pixel_origin,
        y_coordinate_convention=args.y_coordinate_convention,
        screen_space_convention=args.screen_space_convention,
        capture_source=args.capture_source,
        fixed_reference_camera_mode=args.fixed_reference_camera_mode,
        webgpu_camera_constants_source=args.webgpu_camera_constants_source,
        comparison_channel_mode=args.comparison_channel_mode,
        reference_source_kind=args.reference_source_kind,
        webgpu_source_kind=args.webgpu_source_kind,
        conditions_json=Path(args.conditions_json) if args.conditions_json else None,
        include_vertical_flip_diagnostic=not args.disable_vertical_flip_diagnostic,
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
