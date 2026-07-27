#!/usr/bin/env python3
"""Canonical saved-PNG identity and pixel evidence for browser captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import tempfile
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
COLOR_TYPE_NAMES = {
    0: "grayscale",
    2: "rgb",
    4: "grayscale-alpha",
    6: "rgba",
}
COLOR_TYPE_CHANNELS = {0: 1, 2: 3, 4: 2, 6: 4}


def _paeth(left: int, up: int, up_left: int) -> int:
    value = left + up - up_left
    left_distance = abs(value - left)
    up_distance = abs(value - up)
    up_left_distance = abs(value - up_left)
    if left_distance <= up_distance and left_distance <= up_left_distance:
        return left
    return up if up_distance <= up_left_distance else up_left


def _decode_rows(
    compressed: bytes,
    *,
    width: int,
    height: int,
    channels: int,
    bit_depth: int,
) -> List[bytes]:
    if bit_depth != 8:
        raise ValueError(f"unsupported PNG bit depth {bit_depth}")
    row_size = width * channels
    expected_size = height * (row_size + 1)
    if len(compressed) != expected_size:
        raise ValueError(
            f"unexpected decompressed PNG size {len(compressed)}; expected {expected_size}"
        )
    rows: List[bytes] = []
    previous = bytearray(row_size)
    offset = 0
    for _y in range(height):
        filter_type = compressed[offset]
        offset += 1
        encoded = compressed[offset : offset + row_size]
        offset += row_size
        decoded = bytearray(row_size)
        for index, value in enumerate(encoded):
            left = decoded[index - channels] if index >= channels else 0
            up = previous[index]
            up_left = previous[index - channels] if index >= channels else 0
            if filter_type == 0:
                predictor = 0
            elif filter_type == 1:
                predictor = left
            elif filter_type == 2:
                predictor = up
            elif filter_type == 3:
                predictor = (left + up) // 2
            elif filter_type == 4:
                predictor = _paeth(left, up, up_left)
            else:
                raise ValueError(f"unsupported PNG filter type {filter_type}")
            decoded[index] = (value + predictor) & 0xFF
        rows.append(bytes(decoded))
        previous = decoded
    return rows


def inspect_png_file(path: Path) -> Dict[str, Any]:
    resolved = path.resolve()
    result: Dict[str, Any] = {
        "schemaVersion": "phase3-saved-png-pixel-evidence-v1",
        "fileName": path.name,
        "resolvedPath": str(resolved),
        "exists": path.is_file(),
        "sizeBytes": None,
        "modifiedTimeNs": None,
        "sha256": None,
        "width": None,
        "height": None,
        "bitDepth": None,
        "colorType": None,
        "colorTypeName": None,
        "channelCount": None,
        "totalPixelCount": None,
        "rgbNonzeroPixelCount": None,
        "rgbNonblackRatio": None,
        "rgbMax": None,
        "alphaNonzeroPixelCount": None,
        "alphaZeroPixelCount": None,
        "alphaOpaquePixelCount": None,
        "alphaMin": None,
        "alphaMax": None,
        "nonblackBoundingBox": None,
        "pixelClassification": "unknown",
        "decodeError": None,
    }
    if not path.is_file():
        result["decodeError"] = "missing-png"
        return result
    data = path.read_bytes()
    stat = path.stat()
    result.update(
        {
            "sizeBytes": len(data),
            "modifiedTimeNs": stat.st_mtime_ns,
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    )
    try:
        if not data.startswith(PNG_SIGNATURE):
            raise ValueError("not a PNG file")
        offset = len(PNG_SIGNATURE)
        width = height = bit_depth = color_type = None
        idat: List[bytes] = []
        while offset + 12 <= len(data):
            length = struct.unpack(">I", data[offset : offset + 4])[0]
            chunk_type = data[offset + 4 : offset + 8]
            chunk_data = data[offset + 8 : offset + 8 + length]
            offset += 12 + length
            if chunk_type == b"IHDR":
                width, height, bit_depth, color_type = struct.unpack(
                    ">IIBB", chunk_data[:10]
                )
            elif chunk_type == b"IDAT":
                idat.append(chunk_data)
            elif chunk_type == b"IEND":
                break
        if None in (width, height, bit_depth, color_type):
            raise ValueError("missing PNG IHDR")
        channels = COLOR_TYPE_CHANNELS.get(int(color_type))
        if channels is None:
            raise ValueError(f"unsupported PNG color type {color_type}")
        rows = _decode_rows(
            zlib.decompress(b"".join(idat)),
            width=int(width),
            height=int(height),
            channels=channels,
            bit_depth=int(bit_depth),
        )
        rgb_nonzero = 0
        rgb_max = 0
        alpha_nonzero = 0
        alpha_zero = 0
        alpha_opaque = 0
        alpha_min = 255
        alpha_max = 0
        min_x = min_y = max_x = max_y = None
        for y, row in enumerate(rows):
            for x in range(int(width)):
                base = x * channels
                if color_type == 0:
                    red = green = blue = row[base]
                    alpha = 255
                elif color_type == 2:
                    red, green, blue = row[base : base + 3]
                    alpha = 255
                elif color_type == 4:
                    red = green = blue = row[base]
                    alpha = row[base + 1]
                else:
                    red, green, blue, alpha = row[base : base + 4]
                pixel_rgb_max = max(red, green, blue)
                rgb_max = max(rgb_max, pixel_rgb_max)
                alpha_min = min(alpha_min, alpha)
                alpha_max = max(alpha_max, alpha)
                alpha_nonzero += int(alpha > 0)
                alpha_zero += int(alpha == 0)
                alpha_opaque += int(alpha == 255)
                if pixel_rgb_max > 0:
                    rgb_nonzero += 1
                    min_x = x if min_x is None else min(min_x, x)
                    min_y = y if min_y is None else min(min_y, y)
                    max_x = x if max_x is None else max(max_x, x)
                    max_y = y if max_y is None else max(max_y, y)
        pixel_count = int(width) * int(height)
        result.update(
            {
                "width": int(width),
                "height": int(height),
                "bitDepth": int(bit_depth),
                "colorType": int(color_type),
                "colorTypeName": COLOR_TYPE_NAMES.get(int(color_type)),
                "channelCount": channels,
                "totalPixelCount": pixel_count,
                "rgbNonzeroPixelCount": rgb_nonzero,
                "rgbNonblackRatio": rgb_nonzero / pixel_count if pixel_count else None,
                "rgbMax": rgb_max,
                "alphaNonzeroPixelCount": alpha_nonzero,
                "alphaZeroPixelCount": alpha_zero,
                "alphaOpaquePixelCount": alpha_opaque,
                "alphaMin": alpha_min,
                "alphaMax": alpha_max,
                "nonblackBoundingBox": (
                    [min_x, min_y, max_x, max_y] if rgb_nonzero else None
                ),
                "pixelClassification": "nonblank" if rgb_nonzero else "black",
            }
        )
    except Exception as error:  # noqa: BLE001
        result["decodeError"] = str(error)
    return result


def _capture_blob_sha256(status: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(status, dict):
        return None
    identity = status.get("captureBlobIdentity")
    if isinstance(identity, dict) and identity.get("sha256"):
        return str(identity["sha256"])
    value = status.get("blobSha256")
    return str(value) if value else None


def build_saved_png_evidence(
    *,
    base_dir: Path,
    prefix: str,
    capture_status: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    status_file_name = (
        capture_status.get("fileName")
        if isinstance(capture_status, dict)
        else None
    )
    expected_file_name = str(status_file_name or f"{prefix}_canvas.png")
    expected_path = (base_dir / Path(expected_file_name).name).resolve()
    candidates = sorted(
        path.resolve()
        for path in base_dir.glob(f"{prefix}_canvas*.png")
        if path.is_file() and ":" not in path.name
    )
    candidate_paths = [str(path) for path in candidates]
    duplicate = len(candidates) > 1
    ambiguous = len(candidates) != 1 or candidates[0] != expected_path
    file_evidence = inspect_png_file(expected_path)
    blob_sha256 = _capture_blob_sha256(capture_status)
    saved_sha256 = file_evidence.get("sha256")
    identity_match = (
        blob_sha256 == saved_sha256
        if blob_sha256 is not None and saved_sha256 is not None
        else None
    )
    stale = identity_match is False
    exact_file_ready = (
        file_evidence.get("exists") is True
        and file_evidence.get("decodeError") is None
        and not ambiguous
    )
    canonical_result = (
        file_evidence.get("pixelClassification")
        if exact_file_ready and identity_match is True
        else "unknown"
    )
    blocked_reasons = [
        reason
        for reason, failed in [
            ("saved-png-file-missing", file_evidence.get("exists") is not True),
            ("saved-png-file-decode-failed", file_evidence.get("decodeError") is not None),
            ("saved-png-candidate-ambiguous", ambiguous),
            ("capture-blob-hash-missing", blob_sha256 is None),
            ("capture-blob-saved-file-hash-mismatch", identity_match is False),
        ]
        if failed
    ]
    return {
        "schemaVersion": "phase3-saved-png-canonical-evidence-v1",
        "expectedFileName": expected_file_name,
        "expectedResolvedPath": str(expected_path),
        "candidatePaths": candidate_paths,
        "candidateCount": len(candidates),
        "duplicateFileDetected": duplicate,
        "ambiguousFileDetected": ambiguous,
        "staleFileDetected": stale,
        "staleFileState": "stale" if stale else "current" if identity_match is True else "unknown",
        "captureBlobSha256": blob_sha256,
        "savedFileSha256": saved_sha256,
        "blobSavedFileIdentityMatch": identity_match,
        "comparisonFileIsExactSavedFile": not ambiguous and file_evidence.get("exists") is True,
        "savedFilePixelEvidence": file_evidence,
        "savedFilePixelResult": file_evidence.get("pixelClassification"),
        "canonicalSavedPngResult": canonical_result,
        "evidenceDecision": "ready" if not blocked_reasons else "blocked",
        "blockedReasons": blocked_reasons,
    }


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
    )


def _write_rgba_png(path: Path, pixels: Iterable[bytes], width: int, height: int) -> None:
    raw = b"".join(b"\x00" + row for row in pixels)
    path.write_bytes(
        PNG_SIGNATURE
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw))
        + _png_chunk(b"IEND", b"")
    )


def run_self_test() -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="png-pixel-evidence-") as tmp:
        base_dir = Path(tmp)
        prefix = "phase3_step114_fix10_fix2_smoke"
        black = base_dir / f"{prefix}_canvas.png"
        _write_rgba_png(black, [bytes([0, 0, 0, 255] * 2)] * 2, 2, 2)
        black_evidence = inspect_png_file(black)
        black_sha = black_evidence["sha256"]
        identity_evidence = build_saved_png_evidence(
            base_dir=base_dir,
            prefix=prefix,
            capture_status={
                "fileName": black.name,
                "captureBlobIdentity": {"sha256": black_sha},
            },
        )
        nonblank = base_dir / "nonblank.png"
        _write_rgba_png(
            nonblank,
            [bytes([0, 0, 0, 255, 1, 0, 0, 255]), bytes([0, 0, 0, 255] * 2)],
            2,
            2,
        )
        nonblank_evidence = inspect_png_file(nonblank)
        stale_evidence = build_saved_png_evidence(
            base_dir=base_dir,
            prefix=prefix,
            capture_status={
                "fileName": black.name,
                "captureBlobIdentity": {"sha256": nonblank_evidence["sha256"]},
            },
        )
        duplicate = base_dir / f"{prefix}_canvas_1.png"
        shutil.copyfile(black, duplicate)
        duplicate_evidence = build_saved_png_evidence(
            base_dir=base_dir,
            prefix=prefix,
            capture_status={
                "fileName": black.name,
                "captureBlobIdentity": {"sha256": black_sha},
            },
        )
        checks = {
            "blackDetected": black_evidence["pixelClassification"] == "black",
            "alphaOpaqueDoesNotMakeRgbNonblank": black_evidence["alphaOpaquePixelCount"] == 4
            and black_evidence["rgbNonzeroPixelCount"] == 0,
            "nonblankDetected": nonblank_evidence["pixelClassification"] == "nonblank",
            "blobSavedIdentityMatch": identity_evidence["blobSavedFileIdentityMatch"] is True,
            "staleHashMismatchDetected": stale_evidence["staleFileDetected"] is True
            and stale_evidence["canonicalSavedPngResult"] == "unknown",
            "duplicateDetected": duplicate_evidence["duplicateFileDetected"] is True
            and duplicate_evidence["canonicalSavedPngResult"] == "unknown",
        }
        return {
            "status": "ok" if all(checks.values()) else "failed",
            "checks": checks,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = run_self_test()
    elif args.path is not None:
        result = inspect_png_file(args.path)
    else:
        parser.error("path or --self-test is required")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status", "ok") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
