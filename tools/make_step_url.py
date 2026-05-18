#!/usr/bin/env python3
"""
make_step_url.py

Generate deterministic 4DGS Viewer URLs for Step-based GPU candidate testing.

Typical use:
  python3 tools/make_step_url.py \
    --host 100.127.179.39:8080 \
    --source-mode screenCoarse \
    --screen-coarse-max-count 65536 \
    --screen-coarse-min-radius-px 0.25 \
    --screen-coarse-require-in-viewport true \
    --screen-coarse-depth-mode positive

Range example:
  python3 tools/make_step_url.py \
    --host 100.127.179.39:8080 \
    --source-mode range \
    --range-start 0 \
    --range-count 65536

Output:
  A copy-paste-ready URL for 4dgs_gpu_viewer.html.
"""

from __future__ import annotations

import argparse
from urllib.parse import urlencode


DEFAULT_DATASET_TRANSFORM_MATRIX = (
    "0.7071068286895752,0.3535533845424652,-0.6123724579811096,-6.495639801025391,"
    "-0.7071068286895752,0.3535533845424652,-0.6123724579811096,-26.757190704345703,"
    "0,0.8660253882408142,0.5,40.02598571777344,"
    "0,0,0,1"
)


def parse_bool(value: str | bool | None) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "false"

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return "true"
    if text in {"0", "false", "no", "n", "off"}:
        return "false"

    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_base_params(args: argparse.Namespace) -> dict[str, str]:
    return {
        "debugPreserveDrawingBuffer": "1",
        "datasetViewMatrixMode": args.dataset_view_matrix_mode,
        "drawPath": args.draw_path,
        "tileCompositePath": args.tile_composite_path,
        "tileCompositePrimitive": args.tile_composite_primitive,
        "inspectSource": args.inspect_source,
        "inspectJsonMode": args.inspect_json_mode,
        "gpuFramePolicyOverride": args.gpu_frame_policy_override,
        "fixedCanvasWidth": str(args.canvas_width),
        "fixedCanvasHeight": str(args.canvas_height),
        "time": str(args.time),
        "datasetTime": str(args.dataset_time),
        "datasetCameraLabel": args.dataset_camera_label,
        "datasetImageName": args.dataset_image_name,
        "datasetFrameNumber": str(args.dataset_frame_number),
        "datasetViewId": str(args.dataset_view_id),
        "datasetTransformMatrix": args.dataset_transform_matrix,
        "datasetCameraConvention": args.dataset_camera_convention,
        "cameraFoVyRad": str(args.camera_fovy_rad),
        "cameraFoVxRad": str(args.camera_fovx_rad),
        "datasetFx": str(args.dataset_fx),
        "datasetFy": str(args.dataset_fy),
        "datasetCx": str(args.dataset_cx),
        "datasetCy": str(args.dataset_cy),
        "stride": str(args.stride),
        "bgGray": str(args.bg_gray),
        "cudaReferenceLabel": args.cuda_reference_label,
        "useNativeRot4d": parse_bool(args.use_native_rot4d),
        "useNativeMarginal": parse_bool(args.use_native_marginal),
    }


def build_gpu_candidate_params(args: argparse.Namespace) -> dict[str, str]:
    params = {
        "gpuCandidateRuntime": args.runtime,
        "gpuCandidateAllowReadbackInDraw": parse_bool(args.allow_readback_in_draw),
        "gpuCandidatePromotePolicy": args.promote_policy,
        "gpuCandidateCoverageCompare": parse_bool(args.coverage_compare),
        "gpuCandidateCompare": parse_bool(args.candidate_compare),
    }

    if args.source_mode:
        params["gpuCandidateSourceMode"] = args.source_mode

    if args.readback_mode:
        params["gpuCandidateReadbackMode"] = args.readback_mode

    if args.source_mode == "range":
        params.update(
            {
                "gpuCandidateRangeStart": str(args.range_start),
                "gpuCandidateRangeCount": str(args.range_count),
            }
        )

    if args.source_mode == "screenCoarse":
        params.update(
            {
                "gpuCandidateScreenCoarseMaxCount": str(args.screen_coarse_max_count),
                "gpuCandidateScreenCoarseMinRadiusPx": str(args.screen_coarse_min_radius_px),
                "gpuCandidateScreenCoarseRequireInViewport": parse_bool(
                    args.screen_coarse_require_in_viewport
                ),
                "gpuCandidateScreenCoarseDepthMode": args.screen_coarse_depth_mode,
            }
        )

    if args.source_mode == "visibleSrcIndices":
        params.update(
            {
                "gpuCandidateSubsetMode": "visibleSrcIndices",
                "gpuCandidateSubsetCount": str(args.subset_count),
                "gpuCandidateFilterMode": args.filter_mode,
            }
        )

    return params


def build_url(args: argparse.Namespace) -> str:
    host = args.host.rstrip("/")
    viewer_path = args.viewer_path
    if not viewer_path.startswith("/"):
        viewer_path = "/" + viewer_path

    params = build_base_params(args)
    params.update(build_gpu_candidate_params(args))

    if args.extra:
        for item in args.extra:
            if "=" not in item:
                raise ValueError(f"--extra must be key=value, got: {item}")
            key, value = item.split("=", 1)
            params[key] = value

    query = urlencode(params, doseq=False, safe=",:")
    return f"http://{host}{viewer_path}?{query}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic 4DGS Viewer URLs."
    )

    parser.add_argument(
        "--host",
        default="100.127.179.39:8080",
        help="Viewer host:port. Default: 100.127.179.39:8080",
    )
    parser.add_argument(
        "--viewer-path",
        default="/4dgs_gpu_viewer.html",
        help="Viewer HTML path. Default: /4dgs_gpu_viewer.html",
    )

    # Dataset / camera defaults for 000151_v13.
    parser.add_argument("--dataset-view-matrix-mode", default="cuda-aligned")
    parser.add_argument("--draw-path", default="gpu-screen")
    parser.add_argument("--tile-composite-path", default="accumulation")
    parser.add_argument("--tile-composite-primitive", default="quad")
    parser.add_argument("--inspect-source", default="actual-draw")
    parser.add_argument("--inspect-json-mode", default="slim")
    parser.add_argument("--gpu-frame-policy-override", default="auto")
    parser.add_argument("--canvas-width", type=int, default=1280)
    parser.add_argument("--canvas-height", type=int, default=720)
    parser.add_argument("--time", type=float, default=23.2)
    parser.add_argument("--dataset-time", type=float, default=23.2)
    parser.add_argument("--dataset-camera-label", default="000151_v13")
    parser.add_argument("--dataset-image-name", default="000151_v13")
    parser.add_argument("--dataset-frame-number", type=int, default=151)
    parser.add_argument("--dataset-view-id", type=int, default=13)
    parser.add_argument(
        "--dataset-transform-matrix",
        default=DEFAULT_DATASET_TRANSFORM_MATRIX,
    )
    parser.add_argument("--dataset-camera-convention", default="nerf-blender-c2w")
    parser.add_argument("--camera-fovy-rad", type=float, default=0.3995964924806295)
    parser.add_argument("--camera-fovx-rad", type=float, default=0.6911111611634243)
    parser.add_argument("--dataset-fx", type=float, default=1777.7777777777778)
    parser.add_argument("--dataset-fy", type=float, default=1777.7777777777778)
    parser.add_argument("--dataset-cx", type=float, default=640.0)
    parser.add_argument("--dataset-cy", type=float, default=360.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--bg-gray", type=int, default=0)
    parser.add_argument("--cuda-reference-label", default="000151_v13")
    parser.add_argument("--use-native-rot4d", default="true")
    parser.add_argument("--use-native-marginal", default="true")

    # GPU candidate options.
    parser.add_argument("--runtime", default="limited-draw")
    parser.add_argument(
        "--source-mode",
        default="screenCoarse",
        choices=["range", "screenCoarse", "visibleSrcIndices"],
        help="GPU candidate source mode.",
    )
    parser.add_argument("--allow-readback-in-draw", default="true")
    parser.add_argument(
        "--promote-policy",
        default="never",
        choices=["never", "compare-ok", "async-ready", "validated-only"],
        help="GPU candidate promotion policy. Use validated-only for Step110 gated screenCoarse promotion.",
    )
    parser.add_argument("--readback-mode", default="sync-debug")
    parser.add_argument("--coverage-compare", default="true")
    parser.add_argument("--candidate-compare", default="true")

    # Range options.
    parser.add_argument("--range-start", type=int, default=0)
    parser.add_argument("--range-count", type=int, default=65536)

    # screenCoarse options.
    parser.add_argument("--screen-coarse-max-count", type=int, default=65536)
    parser.add_argument("--screen-coarse-min-radius-px", type=float, default=0.25)
    parser.add_argument("--screen-coarse-require-in-viewport", default="true")
    parser.add_argument("--screen-coarse-depth-mode", default="positive")

    # visibleSrcIndices options.
    parser.add_argument("--subset-count", type=int, default=1024)
    parser.add_argument("--filter-mode", default="all-valid")

    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help="Extra query parameter in key=value form. Can be repeated.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(build_url(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
