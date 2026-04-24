#!/usr/bin/env python3
"""Step88-1: extract internal line-like screenshot candidates with OpenCV.

This tool focuses on screenshot-space candidate selection for comparison work.
It prefers internal ridge/line-like regions over outer edges by combining:
- active red/blue foreground masking
- distance transform to background
- structure tensor coherence
- local contrast
- category-specific control maps

Outputs:
- candidate list JSON
- screenshotProbeList string
- annotated accumulation / baseline images
- a score heatmap

The coordinates are screenshot pixels, not Viewer framebuffer/canvas pixels.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np


DEFAULT_ACCUMULATION = Path("/home/demo/work/json/step86_packed_accumulation.png")
DEFAULT_BASELINE = Path("/home/demo/work/json/step86_packed_baseline.png")
DEFAULT_OUTPUT_PREFIX = Path("/home/demo/work/json/step88_internal_candidates_cv2")


@dataclass
class Candidate:
  probeId: str
  category: str
  x: int
  y: int
  score: float
  lineScore: float
  dominance: str
  dominanceDelta: float
  coherence: float
  distanceToBackground: float
  borderDistance: float
  localStd: float
  meanRgb: List[float]
  baselineAccumPatchMeanAbsDiff: float
  selectionBasis: str
  reason: str


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Extract internal line-like screenshot candidates.")
  parser.add_argument("--accumulation", type=Path, default=DEFAULT_ACCUMULATION)
  parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
  parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
  parser.add_argument("--probe-count", type=int, default=8)
  parser.add_argument("--min-spacing", type=int, default=72)
  parser.add_argument("--patch-size", type=int, default=7)
  parser.add_argument("--write-mask", action="store_true")
  return parser.parse_args()


def load_bgr(path: Path) -> np.ndarray:
  img = cv2.imread(str(path), cv2.IMREAD_COLOR)
  if img is None:
    raise FileNotFoundError(f"Unable to read image: {path}")
  return img


def make_active_mask(img_bgr: np.ndarray) -> np.ndarray:
  imgf = img_bgr.astype(np.float32)
  b, g, r = cv2.split(imgf)
  mx = np.maximum.reduce([r, g, b])
  chroma = np.maximum.reduce([np.abs(r - g), np.abs(r - b), np.abs(g - b)])

  mask = ((mx > 35.0) & (chroma > 12.0) & ((r > 45.0) | (b > 45.0))).astype(np.uint8)
  kernel = np.ones((3, 3), np.uint8)
  mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
  mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
  return mask


def structure_tensor_coherence(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)
  gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
  gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
  jxx = cv2.GaussianBlur(gx * gx, (0, 0), 2.0)
  jyy = cv2.GaussianBlur(gy * gy, (0, 0), 2.0)
  jxy = cv2.GaussianBlur(gx * gy, (0, 0), 2.0)
  trace = jxx + jyy
  coherence = np.zeros_like(trace, dtype=np.float32)
  valid = trace > 1e-6
  delta = jxx - jyy
  coherence[valid] = np.sqrt(delta[valid] ** 2 + 4.0 * (jxy[valid] ** 2)) / trace[valid]
  return coherence, jxx, jyy, jxy


def patch_rgb_mean(img_bgr: np.ndarray, x: int, y: int, patch_radius: int) -> Tuple[List[float], float]:
  h, w = img_bgr.shape[:2]
  x0 = max(0, x - patch_radius)
  x1 = min(w, x + patch_radius + 1)
  y0 = max(0, y - patch_radius)
  y1 = min(h, y + patch_radius + 1)
  patch = img_bgr[y0:y1, x0:x1].astype(np.float32)
  if patch.size == 0:
    return [0.0, 0.0, 0.0], 0.0
  mean_bgr = patch.reshape(-1, 3).mean(axis=0)
  # Return RGB order for easier reading in reports.
  mean_rgb = [float(mean_bgr[2]), float(mean_bgr[1]), float(mean_bgr[0])]
  mean_abs_diff = float(np.abs(patch - patch.mean(axis=(0, 1), keepdims=True)).mean())
  return mean_rgb, mean_abs_diff


def patch_mean_abs_diff(a_bgr: np.ndarray, b_bgr: np.ndarray, x: int, y: int, patch_radius: int) -> float:
  h, w = a_bgr.shape[:2]
  x0 = max(0, x - patch_radius)
  x1 = min(w, x + patch_radius + 1)
  y0 = max(0, y - patch_radius)
  y1 = min(h, y + patch_radius + 1)
  pa = a_bgr[y0:y1, x0:x1].astype(np.float32)
  pb = b_bgr[y0:y1, x0:x1].astype(np.float32)
  if pa.size == 0 or pb.size == 0:
    return 0.0
  return float(np.abs(pa - pb).mean())


def nms_peaks(score_map: np.ndarray, min_spacing: int, score_threshold: float) -> List[Tuple[int, int, float]]:
  kernel_size = max(3, int(min_spacing // 2) * 2 + 1)
  kernel = np.ones((kernel_size, kernel_size), np.float32)
  local_max = cv2.dilate(score_map, kernel)
  peak_mask = (score_map >= local_max) & (score_map >= score_threshold)
  ys, xs = np.where(peak_mask)
  peaks = [(int(x), int(y), float(score_map[y, x])) for x, y in zip(xs, ys)]
  peaks.sort(key=lambda p: p[2], reverse=True)
  return peaks


def far_enough(candidate: Candidate, selected: Sequence[Candidate], min_spacing: int) -> bool:
  min_spacing_sq = float(min_spacing * min_spacing)
  for other in selected:
    dx = candidate.x - other.x
    dy = candidate.y - other.y
    if (dx * dx + dy * dy) < min_spacing_sq:
      return False
  return True


def build_maps(img_bgr: np.ndarray, baseline_bgr: np.ndarray) -> Dict[str, np.ndarray]:
  h, w = img_bgr.shape[:2]
  imgf = img_bgr.astype(np.float32)
  b, g, r = cv2.split(imgf)
  mask = make_active_mask(img_bgr)
  mask255 = (mask * 255).astype(np.uint8)

  gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
  coherence, _, _, _ = structure_tensor_coherence(gray)

  dist = cv2.distanceTransform(mask255, cv2.DIST_L2, 5)
  ys, xs = np.indices((h, w))
  border = np.minimum.reduce([xs, ys, w - 1 - xs, h - 1 - ys]).astype(np.float32)

  mean = cv2.blur(gray, (7, 7))
  mean2 = cv2.blur(gray * gray, (7, 7))
  local_std = np.sqrt(np.maximum(mean2 - mean * mean, 0.0))

  # normalize each term to [0, 1]
  coh_n = np.clip((coherence - 0.35) / 0.65, 0.0, 1.0)
  dist_n = np.clip(dist / 12.0, 0.0, 1.0)
  border_n = np.clip(border / 30.0, 0.0, 1.0)
  std_n = np.clip((local_std - 2.0) / 14.0, 0.0, 1.0)

  # line-like core prefers strong coherence and being away from background.
  line_core = (coh_n * dist_n) * mask.astype(np.float32)
  internal_line_map = (0.70 * line_core + 0.20 * std_n + 0.10 * border_n) * mask.astype(np.float32)
  # controls: internal but non-linear, and edge-like line-ish regions.
  internal_control_map = (dist_n * (1.0 - coh_n) * np.maximum(std_n, 0.15)) * mask.astype(np.float32)
  edge_control_map = ((1.0 - dist_n) * np.maximum(coh_n, 0.15)) * mask.astype(np.float32)

  # local mean RGB for dominance classification.
  r_mean = cv2.blur(r, (7, 7))
  b_mean = cv2.blur(b, (7, 7))
  dominance_delta = r_mean - b_mean
  dominance = np.full((h, w), "neutral", dtype=object)
  dominance[dominance_delta > 18.0] = "red"
  dominance[dominance_delta < -18.0] = "blue"

  # Small amount of accumulation-baseline comparison signal to sort ties.
  accum_baseline_abs_diff = np.abs(imgf - baseline_bgr.astype(np.float32))
  accum_baseline_patch_diff = cv2.blur(accum_baseline_abs_diff.mean(axis=2), (7, 7))

  return {
    "mask": mask,
    "dist": dist,
    "border": border,
    "coherence": coherence,
    "coh_n": coh_n,
    "dist_n": dist_n,
    "border_n": border_n,
    "std_n": std_n,
    "dominance_delta": dominance_delta,
    "dominance": dominance,
    "line_core": line_core,
    "internal_line_map": internal_line_map,
    "internal_control_map": internal_control_map,
    "edge_control_map": edge_control_map,
    "baseline_accum_patch_diff": accum_baseline_patch_diff
  }


def make_candidate(
  img_bgr: np.ndarray,
  maps: Dict[str, np.ndarray],
  category: str,
  probe_id: str,
  x: int,
  y: int,
  patch_radius: int,
  selection_basis: str
) -> Candidate:
  line_score = float(maps["line_core"][y, x])
  coherence = float(maps["coherence"][y, x])
  dist = float(maps["dist"][y, x])
  border = float(maps["border"][y, x])
  local_std = float(cv2.blur(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32), (7, 7))[y, x])
  mean_rgb, _ = patch_rgb_mean(img_bgr, x, y, patch_radius)
  # recompute a more stable local std around the pixel from 7x7 window
  y0 = max(0, y - patch_radius)
  y1 = min(img_bgr.shape[0], y + patch_radius + 1)
  x0 = max(0, x - patch_radius)
  x1 = min(img_bgr.shape[1], x + patch_radius + 1)
  patch = img_bgr[y0:y1, x0:x1].astype(np.float32)
  if patch.size > 0:
    gray_patch = cv2.cvtColor(patch.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    local_std = float(gray_patch.std())
  dom_delta = float(maps["dominance_delta"][y, x])
  if dom_delta > 18.0:
    dominance = "red"
  elif dom_delta < -18.0:
    dominance = "blue"
  else:
    dominance = "neutral"
  baseline_diff = float(maps["baseline_accum_patch_diff"][y, x])
  score = float(line_score + 0.18 * np.clip(baseline_diff / 3.0, 0.0, 1.0) + 0.10 * np.clip(local_std / 16.0, 0.0, 1.0))
  if category == "internal-control":
    reason = "internal non-linear control region"
  elif category == "edge-control":
    reason = "edge-like control region"
  elif category.startswith("red"):
    reason = "red-dominant internal line-like region"
  elif category.startswith("blue"):
    reason = "blue-dominant internal line-like region"
  else:
    reason = "internal line-like region"
  return Candidate(
    probeId=probe_id,
    category=category,
    x=int(x),
    y=int(y),
    score=score,
    lineScore=line_score,
    dominance=dominance,
    dominanceDelta=dom_delta,
    coherence=coherence,
    distanceToBackground=dist,
    borderDistance=border,
    localStd=local_std,
    meanRgb=mean_rgb,
    baselineAccumPatchMeanAbsDiff=baseline_diff,
    selectionBasis=selection_basis,
    reason=reason
  )


def select_category_candidates(
  img_bgr: np.ndarray,
  maps: Dict[str, np.ndarray],
  category: str,
  map_name: str,
  probe_prefix: str,
  limit: int,
  min_spacing: int,
  patch_radius: int,
  score_threshold: float
) -> List[Candidate]:
  score_map = maps[map_name]
  peaks = nms_peaks(score_map, min_spacing=min_spacing, score_threshold=score_threshold)
  selected: List[Candidate] = []
  for i, (x, y, _) in enumerate(peaks):
    cand = make_candidate(img_bgr, maps, category, f"{probe_prefix}-{len(selected)+1}", x, y, patch_radius, map_name)
    if not far_enough(cand, selected, min_spacing):
      continue
    selected.append(cand)
    if len(selected) >= limit:
      break
  return selected


def select_internal_candidates(
  img_bgr: np.ndarray,
  maps: Dict[str, np.ndarray],
  map_name: str,
  output_category: str,
  probe_prefix: str,
  limit: int,
  min_spacing: int,
  patch_radius: int,
  dominance_filter: str | None = None
) -> List[Candidate]:
  # Build a candidate pool from internal-line peaks and then filter by dominance.
  internal_candidates = select_category_candidates(
    img_bgr=img_bgr,
    maps=maps,
    category="internal-line",
    map_name=map_name,
    probe_prefix=probe_prefix,
    limit=max(limit * 12, 64),
    min_spacing=max(24, min_spacing // 2),
    patch_radius=patch_radius,
    score_threshold=0.12
  )

  filtered: List[Candidate] = []
  for cand in internal_candidates:
    if dominance_filter == "red" and cand.dominance != "red":
      continue
    if dominance_filter == "blue" and cand.dominance != "blue":
      continue
    if dominance_filter == "neutral" and cand.dominance != "neutral":
      continue
    filtered.append(cand)

  filtered.sort(key=lambda c: (c.score, c.baselineAccumPatchMeanAbsDiff), reverse=True)
  selected: List[Candidate] = []
  for cand in filtered:
    if not far_enough(cand, selected, min_spacing):
      continue
    # Re-label probe IDs to keep category-specific stable names.
    probe_index = len(selected) + 1
    if output_category == "internal-control":
      reason = "internal non-linear control region"
    elif output_category == "edge-control":
      reason = "edge-like control region"
    elif output_category == "blue-internal":
      reason = "blue-dominant internal line-like region"
    else:
      reason = "red-dominant internal line-like region"
    cand = Candidate(**{**asdict(cand), "probeId": f"{probe_prefix}-{probe_index}", "category": output_category})
    cand = Candidate(**{**asdict(cand), "reason": reason})
    selected.append(cand)
    if len(selected) >= limit:
      break
  return selected


def select_edge_controls(
  img_bgr: np.ndarray,
  maps: Dict[str, np.ndarray],
  probe_prefix: str,
  limit: int,
  min_spacing: int,
  patch_radius: int
) -> List[Candidate]:
  candidates = select_category_candidates(
    img_bgr=img_bgr,
    maps=maps,
    category="edge-control",
    map_name="edge_control_map",
    probe_prefix=probe_prefix,
    limit=max(limit * 4, 16),
    min_spacing=max(24, min_spacing // 2),
    patch_radius=patch_radius,
    score_threshold=0.08
  )
  candidates.sort(key=lambda c: (c.score, c.baselineAccumPatchMeanAbsDiff), reverse=True)
  selected: List[Candidate] = []
  for cand in candidates:
    if not far_enough(cand, selected, min_spacing):
      continue
    probe_index = len(selected) + 1
    cand = Candidate(**{**asdict(cand), "probeId": f"{probe_prefix}-{probe_index}", "category": "edge-control"})
    selected.append(cand)
    if len(selected) >= limit:
      break
  return selected


def annotate_candidates(img_bgr: np.ndarray, candidates: Sequence[Candidate]) -> np.ndarray:
  out = img_bgr.copy()
  colors = {
    "red-internal": (0, 0, 255),
    "blue-internal": (255, 0, 0),
    "internal-control": (0, 215, 255),
    "edge-control": (255, 0, 255),
  }
  for idx, cand in enumerate(candidates, start=1):
    color = colors.get(cand.category, (0, 255, 0))
    x, y = cand.x, cand.y
    half = 16
    cv2.rectangle(out, (max(0, x - half), max(0, y - half)), (min(out.shape[1] - 1, x + half), min(out.shape[0] - 1, y + half)), color, 2)
    cv2.circle(out, (x, y), 4, (255, 255, 255), -1)
    label = f"{idx}:{cand.probeId}"
    org = (min(out.shape[1] - 1, x + half + 4), max(16, y - half - 4))
    cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
  return out


def make_heatmap(score_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
  masked = score_map.copy()
  masked[mask == 0] = 0.0
  if np.max(masked) > 0:
    norm = (masked / np.max(masked) * 255.0).astype(np.uint8)
  else:
    norm = masked.astype(np.uint8)
  heat = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
  return heat


def to_jsonable_candidate(c: Candidate) -> Dict[str, Any]:
  return {
    "probeId": c.probeId,
    "category": c.category,
    "x": c.x,
    "y": c.y,
    "score": c.score,
    "lineScore": c.lineScore,
    "dominance": c.dominance,
    "dominanceDelta": c.dominanceDelta,
    "coherence": c.coherence,
    "distanceToBackground": c.distanceToBackground,
    "borderDistance": c.borderDistance,
    "localStd": c.localStd,
    "meanRgb": c.meanRgb,
    "baselineAccumPatchMeanAbsDiff": c.baselineAccumPatchMeanAbsDiff,
    "selectionBasis": c.selectionBasis,
    "reason": c.reason
  }


def candidates_to_probe_list(candidates: Sequence[Candidate]) -> str:
  return ";".join(f"{c.probeId}:{c.x},{c.y}" for c in candidates)


def compute_summary(cands: Sequence[Candidate]) -> Dict[str, Any]:
  return {
    "candidateCount": len(cands),
    "probeIds": [c.probeId for c in cands],
    "categories": [c.category for c in cands],
    "dominanceCounts": {
      "red": sum(1 for c in cands if c.dominance == "red"),
      "blue": sum(1 for c in cands if c.dominance == "blue"),
      "neutral": sum(1 for c in cands if c.dominance == "neutral")
    }
  }


def main() -> int:
  args = parse_args()
  accumulation = load_bgr(args.accumulation)
  baseline = load_bgr(args.baseline)
  if accumulation.shape != baseline.shape:
    raise RuntimeError(f"Image size mismatch: accumulation={accumulation.shape} baseline={baseline.shape}")

  maps = build_maps(accumulation, baseline)
  patch_radius = max(1, int(args.patch_size // 2))
  limit = max(1, int(args.probe_count))
  min_spacing = max(24, int(args.min_spacing))

  # Main candidates: internal line-like, split by dominance.
  red_internal = select_internal_candidates(
    accumulation, maps, "internal_line_map", "red-internal", "red-int", max(2, limit // 3 + 1), min_spacing, patch_radius, dominance_filter="red"
  )
  blue_internal = select_internal_candidates(
    accumulation, maps, "internal_line_map", "blue-internal", "blue-int", max(2, limit // 3 + 1), min_spacing, patch_radius, dominance_filter="blue"
  )
  internal_control = select_internal_candidates(
    accumulation, maps, "internal_control_map", "internal-control", "control-int", 1, min_spacing, patch_radius, dominance_filter=None
  )
  edge_control = select_edge_controls(
    accumulation, maps, "edge-ctrl", 1, min_spacing, patch_radius
  )

  # Assemble final candidate order: line-like internal candidates first, then controls.
  selected: List[Candidate] = []
  for group in [red_internal, blue_internal, internal_control, edge_control]:
    for cand in group:
      if len(selected) >= limit:
        break
      if not far_enough(cand, selected, min_spacing):
        continue
      selected.append(cand)
    if len(selected) >= limit:
      break

  # If category quotas did not fill the list, backfill from the best remaining line-like peaks.
  if len(selected) < limit:
    pool = select_category_candidates(
      img_bgr=accumulation,
      maps=maps,
      category="internal-line",
      map_name="internal_line_map",
      probe_prefix="line",
      limit=max(16, limit * 4),
      min_spacing=max(24, min_spacing // 2),
      patch_radius=patch_radius,
      score_threshold=0.12
    )
    for cand in pool:
      if len(selected) >= limit:
        break
      if not far_enough(cand, selected, min_spacing):
        continue
      selected.append(Candidate(**{**asdict(cand), "probeId": f"line-{len(selected) + 1}"}))

  # Stable final order: internal red, internal blue, internal control, edge control.
  category_order = {"red-internal": 0, "blue-internal": 1, "internal-control": 2, "edge-control": 3}
  selected.sort(key=lambda c: (category_order.get(c.category, 99), -c.score, c.probeId))

  # Build screenshotProbeList string.
  screenshot_probe_list = candidates_to_probe_list(selected)

  # Build an explanatory report.
  report = {
    "cv2Version": cv2.__version__,
    "coordinateSpace": "screenshot-pixels",
    "viewerCoordinateWarning": "These are screenshot pixels, not Viewer framebuffer/canvas coordinates.",
    "inputAccumulation": str(args.accumulation),
    "inputBaseline": str(args.baseline),
    "imageSize": {
      "width": int(accumulation.shape[1]),
      "height": int(accumulation.shape[0]),
    },
    "method": {
      "mask": "maxRGB>35, chroma>12, (red>45 or blue>45) + morph open/close 3x3",
      "linearity": "structure tensor coherence * distance-to-background, with local contrast tie-break",
      "candidatePeak": f"local maxima on score maps with spacing >= {min_spacing}px",
      "patchSize": f"{args.patch_size}x{args.patch_size}",
      "candidateCategories": [
        "red-internal",
        "blue-internal",
        "internal-control",
        "edge-control"
      ],
      "selectionPolicy": {
        "internalLineMap": "prefer high coherence, internal distance, and local contrast",
        "internalControlMap": "prefer internal, low coherence, and non-line-like structure",
        "edgeControlMap": "prefer coherence near the foreground boundary"
      }
    },
    "screenshotProbeList": screenshot_probe_list,
    "candidateSummary": compute_summary(selected),
    "redInternalCandidates": [to_jsonable_candidate(c) for c in red_internal],
    "blueInternalCandidates": [to_jsonable_candidate(c) for c in blue_internal],
    "internalControlCandidates": [to_jsonable_candidate(c) for c in internal_control],
    "edgeControlCandidates": [to_jsonable_candidate(c) for c in edge_control],
    "selectedCandidates": [to_jsonable_candidate(c) for c in selected],
    "outputs": {}
  }

  prefix = args.output_prefix
  prefix.parent.mkdir(parents=True, exist_ok=True)
  json_path = prefix.with_suffix(".json")
  acc_path = prefix.with_name(prefix.name + "_accumulation.png")
  base_path = prefix.with_name(prefix.name + "_baseline.png")
  heat_path = prefix.with_name(prefix.name + "_heatmap.png")
  mask_path = prefix.with_name(prefix.name + "_mask.png")

  report["outputs"] = {
    "json": str(json_path),
    "accumulationAnnotated": str(acc_path),
    "baselineAnnotated": str(base_path),
    "linearityHeatmap": str(heat_path),
  }
  if args.write_mask:
    report["outputs"]["mask"] = str(mask_path)

  # Write report JSON.
  with open(json_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

  # Write visualizations.
  cv2.imwrite(str(acc_path), annotate_candidates(accumulation, selected))
  cv2.imwrite(str(base_path), annotate_candidates(baseline, selected))
  cv2.imwrite(str(heat_path), make_heatmap(maps["internal_line_map"], maps["mask"]))
  if args.write_mask:
    cv2.imwrite(str(mask_path), (maps["mask"] * 255).astype(np.uint8))

  print(json.dumps(report, indent=2, ensure_ascii=False))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
