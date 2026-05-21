#!/usr/bin/env python3
"""
Analyze cpuPostCandidateBreakdown across multiple saved Step prefixes.

Example:
  python3 tools/analyze_cpu_post_candidate_breakdown.py \
    --dir /home/demo/work/json \
    --prefix step125a_000151_v13 \
    --prefix step125a_000056_v08

The tool reads existing JSON files only. It does not depend on jq and does not
touch the viewer runtime.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_PREFIXES = [
    "step125a_000151_v13",
    "step125a_000056_v08",
    "step125a_000083_v16",
    "step125a_000195_v26",
    "step125a_000198_v03",
]


TIMING_FIELDS = [
    "visibleLoopMs",
    "visibleSortMs",
    "packedBuildMs",
    "tileListBuildMs",
    "cpuPostCandidateTotalMs",
]


SCALE_FIELDS = [
    "candidateCount",
    "visibleCount",
    "totalTileRefs",
    "maxRefsPerTile",
    "tileRefsPerVisibleRatio",
]


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {"_value": value}


def get_path(obj: Any, path: str, default: Any = None) -> Any:
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except Exception:
                return default
        else:
            return default
    return cur


def find_key(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = find_key(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_key(value, key)
            if found is not None:
                return found
    return None


def safe_div(numerator: Any, denominator: Any) -> float | None:
    if not isinstance(numerator, (int, float)):
        return None
    if not isinstance(denominator, (int, float)) or denominator == 0:
        return None
    return float(numerator) / float(denominator)


def round_or_none(value: Any, digits: int = 6) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return round(float(value), digits)


def infer_camera(prefix: str) -> str:
    if prefix.startswith("step125a_"):
        return prefix[len("step125a_") :]
    return prefix


def extract_row(base_dir: Path, prefix: str) -> tuple[dict[str, Any] | None, str | None]:
    runtime = load_json(base_dir / f"{prefix}_gpu_candidate_runtime_summary.json")
    limited = load_json(base_dir / f"{prefix}_limited_draw_summary.json")

    breakdown = find_key(runtime, "cpuPostCandidateBreakdown") if runtime else None
    if breakdown is None and limited:
        breakdown = find_key(limited, "cpuPostCandidateBreakdown")
    if not isinstance(breakdown, dict):
        return None, "missing cpuPostCandidateBreakdown"

    timing = breakdown.get("timing", {})
    scale = breakdown.get("scale", {})
    classification = breakdown.get("classification", {})
    debug_readback = bool(
        classification.get("debugReadbackIncluded", breakdown.get("debugReadbackIncluded", False))
    )

    row: dict[str, Any] = {
        "prefix": prefix,
        "camera": infer_camera(prefix),
        "source": breakdown.get("source"),
        "debugReadbackIncluded": debug_readback,
        "dominantStage": classification.get("dominantStage"),
        "webgl2LimitSignals": classification.get("webgl2LimitSignals", []),
        "webgpuMigrationSignals": classification.get("webgpuMigrationSignals", []),
    }

    for key in TIMING_FIELDS:
        row[key] = timing.get(key)
    for key in SCALE_FIELDS:
        row[key] = scale.get(key)

    row["visibleLoopShare"] = safe_div(row.get("visibleLoopMs"), row.get("cpuPostCandidateTotalMs"))
    row["visibleSortShare"] = safe_div(row.get("visibleSortMs"), row.get("cpuPostCandidateTotalMs"))
    row["packedBuildShare"] = safe_div(row.get("packedBuildMs"), row.get("cpuPostCandidateTotalMs"))
    row["tileListBuildShare"] = safe_div(row.get("tileListBuildMs"), row.get("cpuPostCandidateTotalMs"))
    row["visibleLoopMsPerCandidate"] = safe_div(row.get("visibleLoopMs"), row.get("candidateCount"))
    row["visibleLoopMsPerVisible"] = safe_div(row.get("visibleLoopMs"), row.get("visibleCount"))
    row["sortMsPerVisible"] = safe_div(row.get("visibleSortMs"), row.get("visibleCount"))
    row["packedMsPerVisible"] = safe_div(row.get("packedBuildMs"), row.get("visibleCount"))
    row["tileListMsPerTileRef"] = safe_div(row.get("tileListBuildMs"), row.get("totalTileRefs"))
    row["tileRefsPerVisible"] = row.get("tileRefsPerVisibleRatio")

    return row, None


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dominant = Counter(row.get("dominantStage") or "unknown" for row in rows)
    webgl2 = sorted({signal for row in rows for signal in row.get("webgl2LimitSignals", [])})
    webgpu = sorted({signal for row in rows for signal in row.get("webgpuMigrationSignals", [])})

    totals = [row.get("cpuPostCandidateTotalMs") for row in rows if isinstance(row.get("cpuPostCandidateTotalMs"), (int, float))]
    loop_shares = [row.get("visibleLoopShare") for row in rows if isinstance(row.get("visibleLoopShare"), (int, float))]

    return {
        "rowCount": len(rows),
        "dominantStageDistribution": dict(dominant),
        "webgl2LimitSignals": webgl2,
        "webgpuMigrationSignals": webgpu,
        "cpuPostCandidateTotalMsRange": {
            "min": min(totals) if totals else None,
            "max": max(totals) if totals else None,
        },
        "visibleLoopShareRange": {
            "min": min(loop_shares) if loop_shares else None,
            "max": max(loop_shares) if loop_shares else None,
        },
    }


def format_value(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if value != 0 and abs(value) < 0.01:
            return f"{value:.6f}"
        return f"{value:.{digits}f}"
    return str(value)


def print_table(headers: list[str], rows: list[list[Any]]) -> None:
    text_rows = [[format_value(value) for value in row] for row in rows]
    widths = [
        max(len(str(header)), *(len(row[index]) for row in text_rows))
        for index, header in enumerate(headers)
    ]
    print(" | ".join(str(header).ljust(widths[index]) for index, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in text_rows:
        print(" | ".join(row[index].ljust(widths[index]) for index in range(len(headers))))


def print_report(rows: list[dict[str, Any]], skipped: list[dict[str, str]], summary: dict[str, Any]) -> None:
    print("CPU post-candidate breakdown comparison")
    print(f"- rows: {len(rows)}")
    if skipped:
        print(f"- skipped: {len(skipped)}")
        for item in skipped:
            print(f"  - {item['prefix']}: {item['reason']}")
    print()

    print("Timing / scale")
    print_table(
        [
            "prefix",
            "totalMs",
            "visibleLoopMs",
            "sortMs",
            "packedMs",
            "tileListMs",
            "candidate",
            "visible",
            "tileRefs",
            "maxRefsTile",
            "dominant",
        ],
        [
            [
                row["prefix"],
                row.get("cpuPostCandidateTotalMs"),
                row.get("visibleLoopMs"),
                row.get("visibleSortMs"),
                row.get("packedBuildMs"),
                row.get("tileListBuildMs"),
                row.get("candidateCount"),
                row.get("visibleCount"),
                row.get("totalTileRefs"),
                row.get("maxRefsPerTile"),
                row.get("dominantStage"),
            ]
            for row in rows
        ],
    )
    print()

    print("Normalized metrics")
    print_table(
        [
            "prefix",
            "loopShare",
            "sortShare",
            "packedShare",
            "tileShare",
            "loopMs/candidate",
            "loopMs/visible",
            "sortMs/visible",
            "packedMs/visible",
            "tileMs/tileRef",
            "tileRefs/visible",
        ],
        [
            [
                row["prefix"],
                row.get("visibleLoopShare"),
                row.get("visibleSortShare"),
                row.get("packedBuildShare"),
                row.get("tileListBuildShare"),
                row.get("visibleLoopMsPerCandidate"),
                row.get("visibleLoopMsPerVisible"),
                row.get("sortMsPerVisible"),
                row.get("packedMsPerVisible"),
                row.get("tileListMsPerTileRef"),
                row.get("tileRefsPerVisible"),
            ]
            for row in rows
        ],
    )
    print()

    print("Dominant stage distribution")
    for stage, count in summary["dominantStageDistribution"].items():
        print(f"- {stage}: {count}")
    print()

    print("Signal union")
    print("- webgl2LimitSignals:", ", ".join(summary["webgl2LimitSignals"]) or "-")
    print("- webgpuMigrationSignals:", ", ".join(summary["webgpuMigrationSignals"]) or "-")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare cpuPostCandidateBreakdown across saved Step prefixes."
    )
    parser.add_argument("--dir", default="/home/demo/work/json", help="Saved JSON directory.")
    parser.add_argument(
        "--prefix",
        action="append",
        default=[],
        help="Step prefix. Can be repeated. Defaults to the Step125A five-view set.",
    )
    parser.add_argument(
        "--prefix-file",
        default=None,
        help="Optional text file containing one prefix per line.",
    )
    parser.add_argument(
        "--include-debug-readback",
        action="store_true",
        help="Include rows where debugReadbackIncluded=true. Default: exclude.",
    )
    parser.add_argument("--json", default=None, help="Optional output JSON path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = Path(args.dir)
    prefixes = list(args.prefix)
    if args.prefix_file:
        prefixes.extend(
            line.strip()
            for line in Path(args.prefix_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    if not prefixes:
        prefixes = DEFAULT_PREFIXES

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for prefix in prefixes:
        row, reason = extract_row(base_dir, prefix)
        if row is None:
            skipped.append({"prefix": prefix, "reason": reason or "unknown"})
            continue
        if row.get("debugReadbackIncluded") and not args.include_debug_readback:
            skipped.append({"prefix": prefix, "reason": "debugReadbackIncluded=true"})
            continue
        rows.append(row)

    summary = aggregate(rows)
    report = {
        "baseDir": str(base_dir),
        "prefixes": prefixes,
        "rows": rows,
        "skipped": skipped,
        "summary": summary,
    }

    print_report(rows, skipped, summary)

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
