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
import json
import sys
from pathlib import Path
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
    camera_control_contract = args.camera_control_contract
    if args.dataset_view_matrix_mode == "cuda-aligned" and camera_control_contract == "interactive-from-reference":
        camera_control_contract = "reference-fixed"
    params = {
        "debugPreserveDrawingBuffer": parse_bool(args.debug_preserve_drawing_buffer),
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
        "datasetCameraConvention": args.dataset_camera_convention,
        "cameraControlContract": camera_control_contract,
        "cameraOrientationPolicy": args.camera_orientation_policy,
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
    if args.dataset_transform_matrix:
        params["datasetTransformMatrix"] = args.dataset_transform_matrix
    if args.camera_position:
        params["cameraPosition"] = args.camera_position
    if args.camera_target:
        params["cameraTarget"] = args.camera_target
    if args.camera_up:
        params["cameraUp"] = args.camera_up
    return params


def build_gpu_candidate_params(args: argparse.Namespace) -> dict[str, str]:
    params = {
        "gpuCandidateRuntime": args.runtime,
        "gpuCandidateAllowReadbackInDraw": parse_bool(args.allow_readback_in_draw),
        "gpuCandidatePromotePolicy": args.promote_policy,
        "gpuCandidateCoverageCompare": parse_bool(args.coverage_compare),
        "gpuCandidateCompare": parse_bool(args.candidate_compare),
    }
    if parse_bool(args.visible_record_dry_run) == "true":
        params.update(
            {
                "gpuVisibleRecordDryRun": "true",
                "gpuVisibleRecordSource": args.visible_record_source,
                "gpuVisibleRecordReadback": args.visible_record_readback,
                "gpuVisibleRecordMaxCount": str(args.visible_record_max_count),
                "gpuVisibleRecordCompare": parse_bool(args.visible_record_compare),
            }
        )
    if parse_bool(args.raw_visible_record_dry_run) == "true":
        params.update(
            {
                "gpuRawVisibleRecordDryRun": "true",
                "gpuRawVisibleRecordMode": args.raw_visible_record_mode,
                "gpuRawVisibleRecordFields": args.raw_visible_record_fields,
                "gpuRawAttributeTexture": parse_bool(args.raw_attribute_texture),
                "gpuRawVisibleRecordReadback": args.raw_visible_record_readback,
            }
        )
    if parse_bool(args.webgpu_visible_record_dry_run) == "true":
        params.update(
            {
                "webgpuVisibleRecordDryRun": "true",
                "webgpuVisibleRecordMaxCount": str(args.webgpu_visible_record_max_count),
                "webgpuVisibleRecordFields": args.webgpu_visible_record_fields,
            }
        )

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


def flatten_matrix(matrix: object) -> str:
    if not isinstance(matrix, list):
        raise ValueError("camera meta transform_matrix must be a list")

    values: list[float] = []
    for row in matrix:
        if isinstance(row, list):
            values.extend(float(value) for value in row)
        else:
            values.append(float(row))

    if len(values) != 16:
        raise ValueError(f"camera meta transform_matrix must contain 16 values, got {len(values)}")

    return ",".join(str(value) for value in values)


def vector_to_query(values: list[float]) -> str:
    if len(values) < 3:
        raise ValueError("camera vector must contain at least 3 values")
    return ",".join(str(float(values[index])) for index in range(3))


def compute_orbit_pose_from_nerf_blender_c2w(matrix: object) -> tuple[str, str, str]:
    if not isinstance(matrix, list) or len(matrix) < 3:
        raise ValueError("camera meta transform_matrix must contain at least 3 rows")
    rows = []
    for row in matrix[:3]:
        if not isinstance(row, list) or len(row) < 4:
            raise ValueError("camera meta transform_matrix rows must contain at least 4 values")
        rows.append([float(value) for value in row[:4]])

    position = [rows[0][3], rows[1][3], rows[2][3]]
    # NeRF/Blender camera-to-world matrices use local -Z as the viewing
    # direction. For the current CUDA reference camera convention, the
    # roll-free interactive controls axis that preserves the reference image
    # orientation is global -Z: projecting -Z onto the view plane matches the
    # CUDA camera's screen-up direction, while passing the raw camera-local up
    # vector would make OrbitControls inherit camera roll.
    up = [0.0, 0.0, -1.0]
    forward = [-rows[0][2], -rows[1][2], -rows[2][2]]
    target_distance = 10.0
    target = [
        position[0] + forward[0] * target_distance,
        position[1] + forward[1] * target_distance,
        position[2] + forward[2] * target_distance,
    ]
    return vector_to_query(position), vector_to_query(target), vector_to_query(up)


def resolve_camera_meta_path(args: argparse.Namespace) -> Path | None:
    if args.camera_meta_json:
        return Path(args.camera_meta_json)

    if args.camera_name:
        return Path(args.camera_meta_dir) / f"{args.camera_name}_meta.json"

    return None


def apply_camera_meta(args: argparse.Namespace) -> None:
    meta_path = resolve_camera_meta_path(args)
    if meta_path is None:
        return

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    image_name = str(meta["image_name"])

    args.canvas_width = int(meta.get("width", args.canvas_width))
    args.canvas_height = int(meta.get("height", args.canvas_height))
    args.time = float(meta.get("timestamp", args.time))
    args.dataset_time = float(meta.get("timestamp", args.dataset_time))
    args.dataset_camera_label = image_name
    args.dataset_image_name = image_name
    args.dataset_frame_number = int(meta.get("frame_number", args.dataset_frame_number))
    args.dataset_view_id = int(meta.get("view_id", args.dataset_view_id))
    if (
        args.camera_meta_pose_mode == "orbit-initial" and
        args.dataset_view_matrix_mode != "cuda-aligned"
    ):
        position, target, up = compute_orbit_pose_from_nerf_blender_c2w(meta["transform_matrix"])
        args.dataset_transform_matrix = None
        args.camera_position = position
        args.camera_target = target
        args.camera_up = up
    else:
        args.dataset_transform_matrix = flatten_matrix(meta["transform_matrix"])
    args.camera_fovy_rad = float(meta.get("FoVy", args.camera_fovy_rad))
    args.camera_fovx_rad = float(meta.get("FoVx", args.camera_fovx_rad))
    args.dataset_fx = float(meta.get("fx", args.dataset_fx))
    args.dataset_fy = float(meta.get("fy", args.dataset_fy))
    args.dataset_cx = float(meta.get("cx", args.dataset_cx))
    args.dataset_cy = float(meta.get("cy", args.dataset_cy))
    args.cuda_reference_label = image_name


def option_was_provided(argv: list[str], option_names: tuple[str, ...]) -> bool:
    for item in argv[1:]:
        for name in option_names:
            if item == name or item.startswith(name + "="):
                return True
    return False


def set_if_not_provided(
    args: argparse.Namespace,
    argv: list[str],
    dest: str,
    option_names: tuple[str, ...],
    value: object,
) -> None:
    if not option_was_provided(argv, option_names):
        setattr(args, dest, value)


def apply_preset(args: argparse.Namespace, argv: list[str]) -> None:
    if args.preset is None:
        return

    if args.preset == "stable":
        values = {
            "dataset_view_matrix_mode": "threejs",
            "camera_control_contract": "interactive-from-reference",
            "camera_orientation_policy": "roll-free-reference-screen-up",
            "camera_meta_pose_mode": "dataset-transform",
            "debug_preserve_drawing_buffer": "false",
            "runtime": "limited-draw",
            "source_mode": "screenCoarse",
            "promote_policy": "validated-only",
            "allow_readback_in_draw": "false",
            "readback_mode": None,
            "coverage_compare": "false",
            "candidate_compare": "false",
            "visible_record_dry_run": "false",
            "raw_visible_record_dry_run": "false",
            "webgpu_visible_record_dry_run": "false",
        }
    elif args.preset == "validation":
        values = {
            "dataset_view_matrix_mode": "threejs",
            "camera_control_contract": "interactive-from-reference",
            "camera_orientation_policy": "roll-free-reference-screen-up",
            "camera_meta_pose_mode": "dataset-transform",
            "debug_preserve_drawing_buffer": "true",
            "runtime": "limited-draw",
            "source_mode": "screenCoarse",
            "promote_policy": "validated-only",
            "allow_readback_in_draw": "true",
            "readback_mode": "sync-debug",
            "coverage_compare": "true",
            "candidate_compare": "true",
            "visible_record_dry_run": "false",
            "raw_visible_record_dry_run": "true",
            "raw_visible_record_mode": "packed-like",
            "raw_attribute_texture": "true",
            "raw_visible_record_readback": "sync-debug",
            "webgpu_visible_record_dry_run": "false",
        }
    else:
        raise ValueError(f"Unsupported preset: {args.preset}")

    option_names = {
        "dataset_view_matrix_mode": ("--dataset-view-matrix-mode",),
        "camera_control_contract": ("--camera-control-contract",),
        "camera_orientation_policy": ("--camera-orientation-policy",),
        "camera_meta_pose_mode": ("--camera-meta-pose-mode",),
        "dataset_transform_matrix": ("--dataset-transform-matrix",),
        "debug_preserve_drawing_buffer": ("--debug-preserve-drawing-buffer",),
        "runtime": ("--runtime",),
        "source_mode": ("--source-mode",),
        "promote_policy": ("--promote-policy",),
        "allow_readback_in_draw": ("--allow-readback-in-draw",),
        "readback_mode": ("--readback-mode",),
        "coverage_compare": ("--coverage-compare",),
        "candidate_compare": ("--candidate-compare",),
        "visible_record_dry_run": ("--visible-record-dry-run",),
        "raw_visible_record_dry_run": ("--raw-visible-record-dry-run",),
        "raw_visible_record_mode": ("--raw-visible-record-mode",),
        "raw_attribute_texture": ("--raw-attribute-texture",),
        "raw_visible_record_readback": ("--raw-visible-record-readback",),
        "webgpu_visible_record_dry_run": ("--webgpu-visible-record-dry-run",),
    }

    for dest, value in values.items():
        set_if_not_provided(args, argv, dest, option_names[dest], value)


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
    parser.add_argument(
        "--preset",
        choices=["stable", "validation"],
        default=None,
        help="Optional WebGL2 preset. Individual CLI arguments override preset values.",
    )

    # Dataset / camera defaults for 000151_v13.
    parser.add_argument("--dataset-view-matrix-mode", default="cuda-aligned")
    parser.add_argument("--draw-path", default="gpu-screen")
    parser.add_argument("--tile-composite-path", default="accumulation")
    parser.add_argument("--tile-composite-primitive", default="quad")
    parser.add_argument("--debug-preserve-drawing-buffer", default="true")
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
    parser.add_argument("--camera-position", default=None)
    parser.add_argument("--camera-target", default=None)
    parser.add_argument("--camera-up", default=None)
    parser.add_argument("--dataset-camera-convention", default="nerf-blender-c2w")
    parser.add_argument("--camera-control-contract", default="reference-fixed")
    parser.add_argument("--camera-orientation-policy", default="reference-camera-local-up")
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
    parser.add_argument(
        "--camera-meta-json",
        default=None,
        help="Optional CUDA reference camera meta JSON. Overrides dataset camera/time/intrinsics parameters.",
    )
    parser.add_argument(
        "--camera-meta-dir",
        default="/home/demo/work/outputs/sph_scene_4dgs/cuda_reference_named_current_render/iter_012000",
        help="Directory used with --camera-name to locate <camera>_meta.json.",
    )
    parser.add_argument(
        "--camera-name",
        default=None,
        help="Camera/image name such as 000151_v13. Loads <camera>_meta.json from --camera-meta-dir.",
    )
    parser.add_argument(
        "--camera-meta-pose-mode",
        default="dataset-transform",
        choices=["dataset-transform", "orbit-initial"],
        help=(
            "How --camera-name/--camera-meta-json initializes the viewer camera. "
            "dataset-transform preserves CUDA reference matrix params; orbit-initial emits "
            "cameraPosition/cameraTarget/cameraUp for natural OrbitControls interaction."
        ),
    )

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
    parser.add_argument("--visible-record-dry-run", default="false")
    parser.add_argument("--visible-record-source", default="screenCoarse")
    parser.add_argument("--visible-record-readback", default="sync-debug")
    parser.add_argument("--visible-record-max-count", type=int, default=65536)
    parser.add_argument("--visible-record-compare", default="true")
    parser.add_argument("--raw-visible-record-dry-run", default="false")
    parser.add_argument("--raw-visible-record-mode", default="packed-like")
    parser.add_argument("--raw-visible-record-fields", default="srcIndex,valid,px,py,depth,aabb,radius,conic,alpha,tileRange")
    parser.add_argument("--raw-attribute-texture", default="true")
    parser.add_argument("--raw-visible-record-readback", default="sync-debug")
    parser.add_argument("--webgpu-visible-record-dry-run", default="false")
    parser.add_argument("--webgpu-visible-record-max-count", type=int, default=65536)
    parser.add_argument("--webgpu-visible-record-fields", default="srcIndex,valid,px,py,depth,aabb")

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
    apply_preset(args, sys.argv)
    apply_camera_meta(args)
    print(build_url(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
