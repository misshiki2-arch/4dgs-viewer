#!/usr/bin/env python3
"""Step90 camera/projection debug artifacts.

This script intentionally does not tune image overlap.  It compares saved
Viewer debug camera matrices against the CUDA reference meta matrices and
generates visual aids for camera/projection diagnosis.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_JSON_DIR = Path("/home/demo/work/json")
DEFAULT_CUDA_IMAGE = Path(
    "/home/demo/work/outputs/sph_scene_4dgs/cuda_reference_named/iter_012000/000195_v26_render.png"
)
DEFAULT_CUDA_META = Path(
    "/home/demo/work/outputs/sph_scene_4dgs/cuda_reference_named/iter_012000/000195_v26_meta.json"
)
DEFAULT_FORWARD_JSON = DEFAULT_JSON_DIR / "step90_pc1_forward_fix_preserve_debug.json"
DEFAULT_FORWARD_IMAGE = DEFAULT_JSON_DIR / "step90_pc1_forward_fix_preserve_debug.png"
DEFAULT_ALIGNED_JSON = DEFAULT_JSON_DIR / "step90_pc1_cuda_aligned_matrix_debug_fix1.json"
DEFAULT_ALIGNED_IMAGE = DEFAULT_JSON_DIR / "step90_pc1_cuda_aligned_matrix_debug_fix1.png"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def nested_get(obj: Any, path: str, default: Any = None) -> Any:
    cur = obj
    for key in path.split("."):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def to_matrix4(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (4, 4):
        return arr
    if arr.shape == (16,):
        return arr.reshape((4, 4))
    return None


def image_mask(image: Image.Image, threshold: int = 5) -> np.ndarray:
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint16)
    rgb_sum = rgba[..., 0] + rgba[..., 1] + rgba[..., 2]
    return rgb_sum > threshold


def obb_stats(image: Image.Image, label: str) -> dict[str, Any]:
    mask = image_mask(image)
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    if len(xs) == 0:
        return {
            "label": label,
            "width": w,
            "height": h,
            "maskPixelCount": 0,
            "nonBlackRatio": 0.0,
            "aabb": None,
            "centroid": None,
            "principalAxis": None,
            "obb": None,
        }

    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov = np.cov(centered, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    long_axis = vecs[:, 0]
    short_axis = vecs[:, 1]
    # Stabilize orientation for reproducible annotations.
    if long_axis[0] < 0:
        long_axis = -long_axis
        short_axis = -short_axis
    basis = np.stack([long_axis, short_axis], axis=1)
    projected = centered @ basis
    mins = projected.min(axis=0)
    maxs = projected.max(axis=0)
    corners_local = np.array(
        [
            [mins[0], mins[1]],
            [maxs[0], mins[1]],
            [maxs[0], maxs[1]],
            [mins[0], maxs[1]],
        ],
        dtype=np.float64,
    )
    corners = corners_local @ basis.T + centroid
    long_len = float(maxs[0] - mins[0])
    short_len = float(maxs[1] - mins[1])
    angle = float(math.degrees(math.atan2(long_axis[1], long_axis[0])))

    return {
        "label": label,
        "width": w,
        "height": h,
        "maskPixelCount": int(len(xs)),
        "nonBlackRatio": float(len(xs) / (w * h)),
        "aabb": {
            "min": [int(xs.min()), int(ys.min())],
            "maxExclusive": [int(xs.max() + 1), int(ys.max() + 1)],
            "size": [int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
        },
        "centroid": [float(centroid[0]), float(centroid[1])],
        "principalAxis": {
            "long": [float(long_axis[0]), float(long_axis[1])],
            "short": [float(short_axis[0]), float(short_axis[1])],
            "eigenvalues": [float(vals[0]), float(vals[1])],
        },
        "obb": {
            "center": [float(centroid[0]), float(centroid[1])],
            "corners": corners.tolist(),
            "longAxisLength": long_len,
            "shortAxisLength": short_len,
            "angleDeg": angle,
            "longShortRatio": float(long_len / short_len) if short_len > 1e-9 else None,
        },
    }


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
    x, y = xy
    box = draw.textbbox((x, y), text, font=font)
    pad = 8
    draw.rectangle(
        [box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad],
        fill=(0, 0, 0, 190),
    )
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))


def save_side_by_side(images: list[tuple[str, Image.Image]], out_path: Path) -> None:
    width = sum(img.width for _, img in images)
    height = max(img.height for _, img in images)
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    x = 0
    for label, img in images:
        rgb = img.convert("RGB")
        canvas.paste(rgb, (x, 0))
        draw = ImageDraw.Draw(canvas, "RGBA")
        draw_label(draw, (x + 16, 16), label)
        x += img.width
    canvas.save(out_path)


def save_obb_image(image: Image.Image, stats: dict[str, Any], out_path: Path) -> None:
    vis = image.convert("RGB")
    draw = ImageDraw.Draw(vis, "RGBA")
    draw_label(draw, (16, 16), stats["label"])
    if stats.get("aabb"):
        mn = stats["aabb"]["min"]
        mx = stats["aabb"]["maxExclusive"]
        draw.rectangle([mn[0], mn[1], mx[0], mx[1]], outline=(255, 255, 0, 255), width=3)
    if stats.get("obb"):
        corners = [tuple(p) for p in stats["obb"]["corners"]]
        draw.line(corners + [corners[0]], fill=(0, 255, 255, 255), width=3)
        c = stats["obb"]["center"]
        axis = stats["principalAxis"]["long"]
        length = 0.5 * stats["obb"]["longAxisLength"]
        p0 = (c[0] - axis[0] * length, c[1] - axis[1] * length)
        p1 = (c[0] + axis[0] * length, c[1] + axis[1] * length)
        draw.line([p0, p1], fill=(255, 0, 255, 255), width=3)
        draw.ellipse([c[0] - 5, c[1] - 5, c[0] + 5, c[1] + 5], fill=(255, 255, 255, 255))
    vis.save(out_path)


def mat_vec(m: np.ndarray, v: np.ndarray) -> np.ndarray:
    return m @ v


def row_vec_mat(v: np.ndarray, m: np.ndarray) -> np.ndarray:
    return v @ m


PIXEL_CONVENTIONS: dict[str, dict[str, Any]] = {
    "cuda-ndc2pix": {
        "name": "cuda-ndc2pix",
        "yFlip": False,
        "formula": "x = ((ndc_x + 1) * width - 1) * 0.5; y = ((ndc_y + 1) * height - 1) * 0.5",
        "source": "CUDA rasterizer auxiliary.h ndc2Pix(float v, int S)",
    },
    "threejs-y-down": {
        "name": "threejs-y-down",
        "yFlip": True,
        "formula": "x = (ndc_x * 0.5 + 0.5) * width; y = (1 - (ndc_y * 0.5 + 0.5)) * height",
        "source": "Three.js/OpenGL-style NDC to top-left image pixel diagnostic convention",
    },
    "intrinsics-y-down": {
        "name": "intrinsics-y-down",
        "yFlip": False,
        "formula": "x = pixelXSign * fx * (view_x / view_z) + cx; y = fy * (view_y / view_z) + cy",
        "source": "CUDA camera get_rays/image pixel convention using stored fx/fy/cx/cy",
    },
}


def clip_to_pixel(clip: np.ndarray, width: int, height: int, pixel_convention: str) -> dict[str, Any]:
    w = float(clip[3])
    convention = PIXEL_CONVENTIONS[pixel_convention]
    if not np.isfinite(w) or abs(w) < 1e-12:
        return {
            "valid": False,
            "reason": "clip-w-zero",
            "clip": clip.tolist(),
            "pixelConvention": convention,
            "width": width,
            "height": height,
        }
    ndc = clip[:3] / w
    if pixel_convention == "cuda-ndc2pix":
        pixel = np.array([
            ((ndc[0] + 1.0) * width - 1.0) * 0.5,
            ((ndc[1] + 1.0) * height - 1.0) * 0.5,
        ])
    elif pixel_convention == "threejs-y-down":
        pixel = np.array([
            (ndc[0] * 0.5 + 0.5) * width,
            (1.0 - (ndc[1] * 0.5 + 0.5)) * height,
        ])
    else:
        raise ValueError(f"Unsupported clip pixel convention: {pixel_convention}")
    return {
        "valid": bool(np.all(np.isfinite(pixel))),
        "reason": "ok" if np.all(np.isfinite(pixel)) else "non-finite-pixel",
        "clip": clip.tolist(),
        "ndc": ndc.tolist(),
        "pixel": pixel.tolist(),
        "pixelConvention": convention,
        "width": width,
        "height": height,
    }


def project_three(point: np.ndarray, view: np.ndarray, proj: np.ndarray, width: int, height: int) -> dict[str, Any]:
    p = np.array([point[0], point[1], point[2], 1.0], dtype=np.float64)
    view_p = mat_vec(view, p)
    clip = mat_vec(proj, view_p)
    out = clip_to_pixel(clip, width, height, "threejs-y-down")
    out["projectionName"] = "viewer_forward_three"
    out["matrixSource"] = "Viewer actual camera.matrixWorldInverse and camera.projectionMatrix"
    out["view"] = view_p.tolist()
    return out


def project_cuda_full_row(point: np.ndarray, full_proj: np.ndarray, width: int, height: int) -> dict[str, Any]:
    p = np.array([point[0], point[1], point[2], 1.0], dtype=np.float64)
    clip = row_vec_mat(p, full_proj)
    out = clip_to_pixel(clip, width, height, "cuda-ndc2pix")
    out["projectionName"] = "cuda_meta_full_proj_rowvec"
    out["matrixSource"] = "CUDA meta full_proj_transform, applied as row-vector world @ full_proj_transform"
    return out


def project_cuda_intrinsics(
    point: np.ndarray,
    view: np.ndarray,
    intrinsics: dict[str, float],
    pixel_x_sign: float,
) -> dict[str, Any]:
    p = np.array([point[0], point[1], point[2], 1.0], dtype=np.float64)
    view_p = mat_vec(view, p)
    z = float(view_p[2])
    if not np.isfinite(z) or z <= 1e-12:
        return {"valid": False, "reason": "z-not-positive", "view": view_p.tolist()}
    px = pixel_x_sign * intrinsics["fx"] * (view_p[0] / z) + intrinsics["cx"]
    py = intrinsics["fy"] * (view_p[1] / z) + intrinsics["cy"]
    return {
        "valid": bool(np.isfinite(px) and np.isfinite(py)),
        "reason": "ok",
        "projectionName": "intrinsics_projection",
        "matrixSource": "view matrix plus stored fx/fy/cx/cy",
        "pixelConvention": PIXEL_CONVENTIONS["intrinsics-y-down"],
        "width": int(intrinsics.get("width", 0)) if "width" in intrinsics else None,
        "height": int(intrinsics.get("height", 0)) if "height" in intrinsics else None,
        "view": view_p.tolist(),
        "pixel": [float(px), float(py)],
    }


def bounds_debug_points(scene_bounds: dict[str, Any]) -> list[dict[str, Any]]:
    mn = np.asarray(scene_bounds["min"], dtype=np.float64)
    mx = np.asarray(scene_bounds["max"], dtype=np.float64)
    center = np.asarray(scene_bounds["center"], dtype=np.float64)
    size = np.asarray(scene_bounds["size"], dtype=np.float64)
    points: list[dict[str, Any]] = [{"name": "scene_center", "world": center.tolist(), "kind": "center"}]
    for ix in [0, 1]:
        for iy in [0, 1]:
            for iz in [0, 1]:
                p = [mx[0] if ix else mn[0], mx[1] if iy else mn[1], mx[2] if iz else mn[2]]
                points.append({"name": f"bounds_{ix}{iy}{iz}", "world": p, "kind": "bounds_corner"})
    axes = [("x", np.array([1, 0, 0])), ("y", np.array([0, 1, 0])), ("z", np.array([0, 0, 1]))]
    for name, axis in axes:
        half = 0.5 * size[{"x": 0, "y": 1, "z": 2}[name]]
        points.append({"name": f"axis_{name}_minus", "world": (center - axis * half).tolist(), "kind": "bounds_axis"})
        points.append({"name": f"axis_{name}_plus", "world": (center + axis * half).tolist(), "kind": "bounds_axis"})
    return points


def pixel_delta(a: dict[str, Any], b: dict[str, Any]) -> list[float] | None:
    if not a.get("valid") or not b.get("valid"):
        return None
    return [float(b["pixel"][0] - a["pixel"][0]), float(b["pixel"][1] - a["pixel"][1])]


def summarize_deltas(points: list[dict[str, Any]], source: str, target: str) -> dict[str, Any]:
    deltas = []
    for p in points:
        d = p.get("pixelDeltas", {}).get(f"{source}_to_{target}")
        if d is not None:
            deltas.append(d)
    if not deltas:
        return {"count": 0}
    arr = np.asarray(deltas, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1)
    return {
        "count": int(len(deltas)),
        "meanDelta": arr.mean(axis=0).tolist(),
        "medianDelta": np.median(arr, axis=0).tolist(),
        "meanDistance": float(norms.mean()),
        "maxDistance": float(norms.max()),
    }


def draw_projection_overlay(base: Image.Image, label: str, points: list[dict[str, Any]], projection_key: str, out_path: Path) -> None:
    vis = base.convert("RGB")
    draw = ImageDraw.Draw(vis, "RGBA")
    draw_label(draw, (16, 16), label)
    colors = {
        "center": (255, 255, 255, 255),
        "bounds_corner": (255, 255, 0, 230),
        "bounds_axis": (0, 255, 255, 230),
    }
    for p in points:
        proj = p["projections"].get(projection_key)
        if not proj or not proj.get("valid"):
            continue
        x, y = proj["pixel"]
        color = colors.get(p.get("kind"), (255, 255, 255, 220))
        r = 6 if p["kind"] == "center" else 4
        draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=2)
        if p["kind"] == "center":
            draw.text((x + 8, y + 8), p["name"], fill=color)
    vis.save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_JSON_DIR)
    parser.add_argument("--cuda-image", type=Path, default=DEFAULT_CUDA_IMAGE)
    parser.add_argument("--cuda-meta", type=Path, default=DEFAULT_CUDA_META)
    parser.add_argument("--forward-json", type=Path, default=DEFAULT_FORWARD_JSON)
    parser.add_argument("--forward-image", type=Path, default=DEFAULT_FORWARD_IMAGE)
    parser.add_argument("--aligned-json", type=Path, default=DEFAULT_ALIGNED_JSON)
    parser.add_argument("--aligned-image", type=Path, default=DEFAULT_ALIGNED_IMAGE)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cuda_img = Image.open(args.cuda_image)
    forward_img = Image.open(args.forward_image)
    aligned_img = Image.open(args.aligned_image)
    meta = load_json(args.cuda_meta)
    forward = load_json(args.forward_json)
    aligned = load_json(args.aligned_json)

    save_side_by_side(
        [
            ("CUDA Reference", cuda_img),
            ("Viewer forward-fix", forward_img),
            ("Viewer cuda-aligned", aligned_img),
        ],
        args.out_dir / "step90_side_by_side_cuda_forward_cudaaligned.png",
    )

    images = [
        ("CUDA Reference", cuda_img, "step90_obb_cuda_reference.png"),
        ("Viewer forward-fix", forward_img, "step90_obb_forward_fix.png"),
        ("Viewer cuda-aligned", aligned_img, "step90_obb_cuda_aligned_fix1.png"),
    ]
    obb = {}
    obb_images = []
    for label, image, filename in images:
        stats = obb_stats(image, label)
        obb[label] = stats
        out = args.out_dir / filename
        save_obb_image(image, stats, out)
        obb_images.append((label, Image.open(out)))
    save_side_by_side(obb_images, args.out_dir / "step90_obb_side_by_side.png")

    scene_bounds = nested_get(forward, "cameraDebugState.sceneBounds") or nested_get(
        aligned, "cameraDebugState.sceneBounds"
    )
    if not scene_bounds:
        raise RuntimeError("sceneBounds missing from debug JSON")

    width = int(meta["width"])
    height = int(meta["height"])
    full_proj = np.asarray(meta["full_proj_transform"], dtype=np.float64)
    meta_view_row = np.asarray(meta["world_view_transform"], dtype=np.float64).T
    forward_view = to_matrix4(nested_get(forward, "cameraDebugState.camera.matrixWorldInverse"))
    forward_proj = to_matrix4(nested_get(forward, "cameraDebugState.camera.projectionMatrix"))
    aligned_summary = nested_get(aligned, "bundle.deterministicState.cudaAlignedScreenSpaceCamera") or nested_get(
        aligned, "cameraDebugState.deterministicState.cudaAlignedScreenSpaceCamera"
    )
    aligned_view = to_matrix4(aligned_summary.get("cudaAlignedViewMatrix")) if aligned_summary else None
    intrinsics = {
        "fx": float(meta["fx"]),
        "fy": float(meta["fy"]),
        "cx": float(meta["cx"]),
        "cy": float(meta["cy"]),
        "width": width,
        "height": height,
    }
    pixel_x_sign = float(aligned_summary.get("pixelXSign", 1)) if aligned_summary else 1.0

    debug_points = bounds_debug_points(scene_bounds)
    for p in debug_points:
        world = np.asarray(p["world"], dtype=np.float64)
        projections: dict[str, Any] = {
            "cuda_meta_full_proj_rowvec": project_cuda_full_row(world, full_proj, width, height),
            "cuda_meta_view_intrinsics": project_cuda_intrinsics(world, meta_view_row, intrinsics, 1.0),
        }
        projections["cuda_meta_view_intrinsics"]["projectionName"] = "cuda_meta_view_intrinsics"
        projections["cuda_meta_view_intrinsics"][
            "matrixSource"
        ] = "CUDA meta world_view_transform transpose plus stored fx/fy/cx/cy"
        if forward_view is not None and forward_proj is not None:
            projections["viewer_forward_three"] = project_three(world, forward_view, forward_proj, width, height)
        else:
            projections["viewer_forward_three"] = {
                "valid": False,
                "reason": "forward-camera-matrix-missing",
                "projectionName": "viewer_forward_three",
                "matrixSource": "Viewer actual camera matrices",
                "pixelConvention": PIXEL_CONVENTIONS["threejs-y-down"],
                "width": width,
                "height": height,
            }
        if aligned_view is not None:
            projections["viewer_cuda_aligned_intrinsics"] = project_cuda_intrinsics(
                world, aligned_view, intrinsics, pixel_x_sign
            )
            projections["viewer_cuda_aligned_intrinsics"]["projectionName"] = "viewer_cuda_aligned_intrinsics"
            projections["viewer_cuda_aligned_intrinsics"][
                "matrixSource"
            ] = "Viewer cudaAlignedViewMatrix plus stored fx/fy/cx/cy"
        else:
            projections["viewer_cuda_aligned_intrinsics"] = {
                "valid": False,
                "reason": "cuda-aligned-view-matrix-missing",
                "projectionName": "viewer_cuda_aligned_intrinsics",
                "matrixSource": "Viewer cudaAlignedViewMatrix plus stored fx/fy/cx/cy",
                "pixelConvention": PIXEL_CONVENTIONS["intrinsics-y-down"],
                "width": width,
                "height": height,
            }
        p["projections"] = projections
        p["pixelDeltas"] = {
            "cuda_meta_full_proj_rowvec_to_cuda_meta_view_intrinsics": pixel_delta(
                projections["cuda_meta_full_proj_rowvec"], projections["cuda_meta_view_intrinsics"]
            ),
            "cuda_meta_full_proj_rowvec_to_viewer_forward_three": pixel_delta(
                projections["cuda_meta_full_proj_rowvec"], projections["viewer_forward_three"]
            ),
            "cuda_meta_view_intrinsics_to_viewer_forward_three": pixel_delta(
                projections["cuda_meta_view_intrinsics"], projections["viewer_forward_three"]
            ),
            "cuda_meta_view_intrinsics_to_viewer_cuda_aligned_intrinsics": pixel_delta(
                projections["cuda_meta_view_intrinsics"], projections["viewer_cuda_aligned_intrinsics"]
            ),
            "cuda_meta_full_proj_rowvec_to_viewer_cuda_aligned_intrinsics": pixel_delta(
                projections["cuda_meta_full_proj_rowvec"], projections["viewer_cuda_aligned_intrinsics"]
            ),
        }

    matrix_diagnostics = {
        "note": "CUDA world_view_transform/full_proj_transform are stored in the torch/rasterizer orientation; row-major projection variants are diagnostic, not an image-quality score.",
        "pixelConventions": PIXEL_CONVENTIONS,
        "projectionDefinitions": {
            "cuda_meta_full_proj_rowvec": {
                "matrixSource": "CUDA meta full_proj_transform",
                "matrixApplication": "row-vector world coordinate @ full_proj_transform",
                "pixelConvention": "cuda-ndc2pix",
                "expectedCudaRasterizerMatch": True,
            },
            "cuda_meta_view_intrinsics": {
                "matrixSource": "transpose(CUDA meta world_view_transform) plus fx/fy/cx/cy",
                "matrixApplication": "row-major view matrix @ world coordinate, then intrinsics",
                "pixelConvention": "intrinsics-y-down",
                "expectedCudaRasterizerMatch": True,
            },
            "viewer_forward_three": {
                "matrixSource": "Viewer actual Three.js camera matrices",
                "matrixApplication": "projectionMatrix @ matrixWorldInverse @ world coordinate",
                "pixelConvention": "threejs-y-down",
                "expectedCudaRasterizerMatch": False,
            },
            "viewer_cuda_aligned_intrinsics": {
                "matrixSource": "Viewer cudaAlignedViewMatrix plus fx/fy/cx/cy",
                "matrixApplication": "row-major view matrix @ world coordinate, then intrinsics",
                "pixelConvention": "intrinsics-y-down",
                "expectedCudaRasterizerMatch": True,
            },
        },
        "metaWorldViewTransformRaw": meta["world_view_transform"],
        "metaWorldViewTransformTransposeUsedAsRowMajor": meta_view_row.tolist(),
        "metaFullProjTransformRaw": meta["full_proj_transform"],
        "forwardViewMatrix": forward_view.tolist() if forward_view is not None else None,
        "forwardProjectionMatrix": forward_proj.tolist() if forward_proj is not None else None,
        "cudaAlignedViewMatrix": aligned_view.tolist() if aligned_view is not None else None,
        "cudaAlignedPixelXSign": pixel_x_sign,
        "projectionDeltaSummaries": {
            "cudaFull_to_cudaViewIntrinsics": summarize_deltas(
                debug_points, "cuda_meta_full_proj_rowvec", "cuda_meta_view_intrinsics"
            ),
            "cudaFull_to_forwardThree": summarize_deltas(
                debug_points, "cuda_meta_full_proj_rowvec", "viewer_forward_three"
            ),
            "cudaViewIntrinsics_to_forwardThree": summarize_deltas(
                debug_points, "cuda_meta_view_intrinsics", "viewer_forward_three"
            ),
            "cudaViewIntrinsics_to_cudaAlignedIntrinsics": summarize_deltas(
                debug_points, "cuda_meta_view_intrinsics", "viewer_cuda_aligned_intrinsics"
            ),
            "cudaFull_to_cudaAlignedIntrinsics": summarize_deltas(
                debug_points, "cuda_meta_full_proj_rowvec", "viewer_cuda_aligned_intrinsics"
            ),
        },
    }

    result = {
        "inputs": {
            "cudaImage": str(args.cuda_image),
            "cudaMeta": str(args.cuda_meta),
            "forwardJson": str(args.forward_json),
            "forwardImage": str(args.forward_image),
            "alignedJson": str(args.aligned_json),
            "alignedImage": str(args.aligned_image),
        },
        "cudaMetaSummary": {
            key: meta.get(key)
            for key in [
                "image_name",
                "frame_number",
                "view_id",
                "timestamp",
                "width",
                "height",
                "FoVx",
                "FoVy",
                "fx",
                "fy",
                "cx",
                "cy",
                "camera_center",
                "transform_matrix",
            ]
        },
        "sceneBounds": scene_bounds,
        "obb": obb,
        "projectionDebugPoints": debug_points,
        "matrixDiagnostics": matrix_diagnostics,
        "representativeSplats": {
            "available": False,
            "reason": "The saved debug JSON contains scene bounds/camera summaries, but not raw per-splat world coordinates for non-black contributing splats.",
        },
    }
    points_path = args.out_dir / "step90_projection_debug_points.json"
    points_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    draw_projection_overlay(
        cuda_img,
        "CUDA meta full-proj points",
        debug_points,
        "cuda_meta_full_proj_rowvec",
        args.out_dir / "step90_projection_debug_overlay_cuda.png",
    )
    draw_projection_overlay(
        forward_img,
        "Viewer forward three.js points",
        debug_points,
        "viewer_forward_three",
        args.out_dir / "step90_projection_debug_overlay_forward_fix.png",
    )
    draw_projection_overlay(
        aligned_img,
        "Viewer cuda-aligned intrinsics points",
        debug_points,
        "viewer_cuda_aligned_intrinsics",
        args.out_dir / "step90_projection_debug_overlay_cuda_aligned.png",
    )
    save_side_by_side(
        [
            ("CUDA meta full-proj", Image.open(args.out_dir / "step90_projection_debug_overlay_cuda.png")),
            ("Viewer forward projection", Image.open(args.out_dir / "step90_projection_debug_overlay_forward_fix.png")),
            (
                "Viewer cuda-aligned projection",
                Image.open(args.out_dir / "step90_projection_debug_overlay_cuda_aligned.png"),
            ),
        ],
        args.out_dir / "step90_projection_debug_side_by_side.png",
    )

    print(json.dumps({
        "sideBySide": str(args.out_dir / "step90_side_by_side_cuda_forward_cudaaligned.png"),
        "obbSideBySide": str(args.out_dir / "step90_obb_side_by_side.png"),
        "projectionPoints": str(points_path),
        "projectionSideBySide": str(args.out_dir / "step90_projection_debug_side_by_side.png"),
        "obbSummary": {
            label: {
                "maskPixelCount": stats["maskPixelCount"],
                "nonBlackRatio": stats["nonBlackRatio"],
                "aabb": stats["aabb"],
                "centroid": stats["centroid"],
                "obb": stats["obb"],
            }
            for label, stats in obb.items()
        },
        "projectionDeltaSummaries": matrix_diagnostics["projectionDeltaSummaries"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
