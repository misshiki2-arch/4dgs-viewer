#!/usr/bin/env python3
"""Focused numeric regression for compare_png comparison channel modes."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from compare_png import (
    build_absdiff_image,
    compare_images,
    compute_pixel_metrics,
)


COMPARE_TOOL = Path(__file__).with_name("compare_png.py")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_close(actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"expected {expected}, got {actual}")


def _save_rgba(path: Path, values: np.ndarray) -> None:
    Image.fromarray(values, mode="RGBA").save(path)


def _compare_fixture(a_path: Path, b_path: Path, mode: str) -> dict:
    return compare_images(
        a_path=a_path,
        b_path=b_path,
        nonblack_threshold=0,
        diff_scale=1.0,
        out_path=None,
        reference_source="channel-mode-fixture-a",
        webgpu_source="channel-mode-fixture-b",
        camera_label="fixture-camera",
        frame_label="fixture-frame",
        dataset_time="0",
        viewport="1x2",
        background_policy="black",
        color_space_policy="encoded-rgb",
        pixel_origin="top-left-saved-png",
        y_coordinate_convention="screen-y-down",
        screen_space_convention="top-left-y-down",
        capture_source="fixture",
        fixed_reference_camera_mode="enabled",
        webgpu_camera_constants_source="fixture",
        comparison_channel_mode=mode,
        reference_source_kind="fixture",
        webgpu_source_kind="fixture",
        conditions_json=None,
        include_vertical_flip_diagnostic=True,
    )


def main() -> int:
    rgb_a = np.array([[[0, 0, 0, 10], [10, 20, 30, 40]]], dtype=np.uint8)
    rgb_b = np.array([[[3, 4, 0, 10], [10, 25, 35, 40]]], dtype=np.uint8)

    rgb_metrics = compute_pixel_metrics(rgb_a, rgb_b, "rgb")
    assert rgb_metrics["comparisonChannelCount"] == 3
    assert rgb_metrics["valueCount"] == 6
    _assert_close(rgb_metrics["mae"], 17.0 / 6.0)
    _assert_close(rgb_metrics["rmse"], math.sqrt(75.0 / 6.0))
    assert rgb_metrics["maxAbsError"] == 5
    assert rgb_metrics["sumAbsError"] == 17
    assert rgb_metrics["differentPixelCountAnyChannel"] == 2
    assert rgb_metrics["differentPixelCountAllChannels"] == 0
    assert rgb_metrics["differentValueCount"] == 4
    assert rgb_metrics["samePixels"] == 0
    _assert_close(rgb_metrics["differentPixelRatioAnyChannel"], 1.0)

    rgba_metrics = compute_pixel_metrics(rgb_a, rgb_b, "rgba")
    default_metrics = compute_pixel_metrics(rgb_a, rgb_b)
    assert rgba_metrics == default_metrics
    assert rgba_metrics["comparisonChannelCount"] == 4
    assert rgba_metrics["valueCount"] == 8
    _assert_close(rgba_metrics["mae"], 17.0 / 8.0)
    _assert_close(rgba_metrics["rmse"], math.sqrt(75.0 / 8.0))

    alpha_a = np.array([[[7, 8, 9, 10]]], dtype=np.uint8)
    alpha_b = np.array([[[7, 8, 9, 11]]], dtype=np.uint8)
    alpha_rgb = compute_pixel_metrics(alpha_a, alpha_b, "rgb")
    alpha_rgba = compute_pixel_metrics(alpha_a, alpha_b, "rgba")
    assert alpha_rgb["mae"] == 0.0
    assert alpha_rgb["rmse"] == 0.0
    assert alpha_rgb["differentPixelCountAnyChannel"] == 0
    assert alpha_rgb["differentValueCount"] == 0
    assert alpha_rgba["sumAbsError"] == 1
    assert alpha_rgba["differentPixelCountAnyChannel"] == 1
    assert alpha_rgba["differentValueCount"] == 1

    try:
        compute_pixel_metrics(rgb_a, rgb_b, "grayscale")
    except ValueError as error:
        assert "unsupported comparison channel mode" in str(error)
    else:
        raise AssertionError("unsupported direct-call mode was accepted")

    alpha_diff = np.abs(alpha_a.astype(np.int16) - alpha_b.astype(np.int16)).astype(
        np.uint8
    )
    absdiff_pixels = np.asarray(build_absdiff_image(alpha_diff, 1.0))
    assert np.all(absdiff_pixels[..., :3] == 0)
    assert np.all(absdiff_pixels[..., 3] == 255)

    with tempfile.TemporaryDirectory(prefix="compare-png-channel-mode-") as tmp:
        tmp_path = Path(tmp)
        orientation_a = np.array(
            [[[10, 0, 0, 10]], [[20, 0, 0, 20]]], dtype=np.uint8
        )
        orientation_b = np.array(
            [[[20, 0, 0, 10]], [[10, 0, 0, 20]]], dtype=np.uint8
        )
        a_path = tmp_path / "a.png"
        b_path = tmp_path / "b.png"
        _save_rgba(a_path, orientation_a)
        _save_rgba(b_path, orientation_b)
        alpha_a_path = tmp_path / "alpha-a.png"
        alpha_b_path = tmp_path / "alpha-b.png"
        _save_rgba(alpha_a_path, alpha_a)
        _save_rgba(alpha_b_path, alpha_b)
        input_hashes = tuple(
            _sha256(path)
            for path in [a_path, b_path, alpha_a_path, alpha_b_path]
        )

        rgb_summary = _compare_fixture(a_path, b_path, "rgb")
        rgba_summary = _compare_fixture(a_path, b_path, "rgba")
        assert rgb_summary["channelCount"] == 4
        assert rgb_summary["loadedChannelCount"] == 4
        assert rgb_summary["loadedValueCount"] == 8
        assert rgb_summary["comparisonChannelCount"] == 3
        assert rgb_summary["valueCount"] == 6
        assert rgb_summary["orientationDiagnostic"]["normal"][
            "comparisonChannelMode"
        ] == "rgb"
        assert rgb_summary["orientationDiagnostic"]["verticalFlip"][
            "comparisonChannelCount"
        ] == 3
        assert rgb_summary["orientationDiagnostic"]["verticalFlip"]["rmse"] == 0.0
        assert rgba_summary["orientationDiagnostic"]["verticalFlip"][
            "comparisonChannelCount"
        ] == 4
        assert rgba_summary["orientationDiagnostic"]["verticalFlip"]["rmse"] > 0.0

        alpha_rgb_summary = _compare_fixture(alpha_a_path, alpha_b_path, "rgb")
        alpha_rgba_summary = _compare_fixture(alpha_a_path, alpha_b_path, "rgba")
        assert alpha_rgb_summary["differentPixelRatioAnyChannel"] == 0.0
        assert alpha_rgb_summary["visualMismatchClassification"] == "parity-candidate"
        assert alpha_rgba_summary["differentPixelRatioAnyChannel"] == 1.0
        assert (
            alpha_rgba_summary["visualMismatchClassification"]
            == "comparison-ready-difference-unclassified"
        )

        default_json = tmp_path / "default.json"
        explicit_json = tmp_path / "explicit-rgba.json"
        common_command = [
            sys.executable,
            str(COMPARE_TOOL),
            "--a",
            str(a_path),
            "--b",
            str(b_path),
        ]
        subprocess.run(
            [*common_command, "--json", str(default_json)],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                *common_command,
                "--comparison-channel-mode",
                "rgba",
                "--json",
                str(explicit_json),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        default_summary = json.loads(default_json.read_text(encoding="utf-8"))
        explicit_summary = json.loads(explicit_json.read_text(encoding="utf-8"))
        compared_fields = [
            "comparisonChannelMode",
            "comparisonChannelCount",
            "valueCount",
            "mae",
            "rmse",
            "maxAbsError",
            "sumAbsError",
            "differentPixelCountAnyChannel",
            "differentPixelCountAllChannels",
            "differentValueCount",
            "samePixels",
            "differentPixelRatioAnyChannel",
            "visualMismatchClassification",
            "orientationDiagnostic",
        ]
        assert {
            field: default_summary[field] for field in compared_fields
        } == {field: explicit_summary[field] for field in compared_fields}

        invalid = subprocess.run(
            [*common_command, "--comparison-channel-mode", "grayscale"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert invalid.returncode != 0
        assert "invalid choice" in invalid.stderr
        assert input_hashes == tuple(
            _sha256(path)
            for path in [a_path, b_path, alpha_a_path, alpha_b_path]
        )

    print("compare_png channel-mode smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
