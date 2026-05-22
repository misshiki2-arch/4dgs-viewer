#!/usr/bin/env python3
"""
check_step_files.py

Check saved Step JSON/PNG files.

Typical use:
  python3 tools/check_step_files.py \
    --dir /home/demo/work/json \
    --prefix step107_000151_v13

Optional:
  python3 tools/check_step_files.py \
    --dir /home/demo/work/json \
    --prefix step107_000151_v13 \
    --json /home/demo/work/json/step107_000151_v13_file_check_summary.json

Purpose:
- Check required JSON/PNG files for a given step prefix.
- Detect accidental ".json.png" files.
- Validate JSON parseability.
- Report file sizes and missing files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_JSON_SUFFIXES = [
    "gpu_candidate_coverage",
    "gpu_candidate_runtime_summary",
    "limited_draw_summary",
    "visible_compare",
    "live_same_state",
    "association",
    "summary",
]

DEFAULT_JSON_ONE_OF_GROUPS = [
    [
        "gpu_candidate_screen_coarse_compare",
        "gpu_candidate_source_compare",
    ],
]

DEFAULT_OPTIONAL_JSON_SUFFIXES = [
    "gpu_candidate_screen_coarse_dryrun_visible_compare",
    "gpu_visible_record_dryrun_compare",
    "gpu_raw_visible_record_dryrun_compare",
    "webgpu_visible_record_dryrun_compare",
]

DEFAULT_PNG_SUFFIXES = [
    "canvas",
]


def file_size_human(size: int) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{size} B"


def parse_csv_list(value: str | None) -> List[str]:
    if value is None or value.strip() == "":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def validate_json(path: Path) -> Tuple[bool, str | None]:
    try:
        with path.open("r", encoding="utf-8") as f:
            json.load(f)
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def check_expected_files(
    base_dir: Path,
    prefix: str,
    json_suffixes: List[str],
    png_suffixes: List[str],
    json_one_of_groups: List[List[str]] | None = None,
    optional_json_suffixes: List[str] | None = None,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {
        "baseDir": str(base_dir),
        "prefix": prefix,
        "jsonChecks": [],
        "jsonOneOfChecks": [],
        "optionalJsonChecks": [],
        "pngChecks": [],
        "unexpectedJsonPngFiles": [],
        "allRequiredPresent": True,
        "allJsonValid": True,
        "hasJsonPngProblem": False,
    }

    for suffix in json_suffixes:
        path = base_dir / f"{prefix}_{suffix}.json"
        entry: Dict[str, Any] = {
            "suffix": suffix,
            "path": str(path),
            "exists": path.exists(),
            "sizeBytes": None,
            "sizeHuman": None,
            "jsonValid": None,
            "error": None,
        }

        if not path.exists():
            results["allRequiredPresent"] = False
            results["allJsonValid"] = False
            entry["jsonValid"] = False
            entry["error"] = "missing"
        else:
            size = path.stat().st_size
            entry["sizeBytes"] = size
            entry["sizeHuman"] = file_size_human(size)

            if size == 0:
                results["allJsonValid"] = False
                entry["jsonValid"] = False
                entry["error"] = "empty file"
            else:
                valid, error = validate_json(path)
                entry["jsonValid"] = valid
                entry["error"] = error
                if not valid:
                    results["allJsonValid"] = False

        results["jsonChecks"].append(entry)

    for group in json_one_of_groups or []:
        group_entries = []
        present_valid_suffixes = []
        present_suffixes = []

        for suffix in group:
            path = base_dir / f"{prefix}_{suffix}.json"
            entry = {
                "suffix": suffix,
                "path": str(path),
                "exists": path.exists(),
                "sizeBytes": None,
                "sizeHuman": None,
                "jsonValid": None,
                "error": None,
            }

            if path.exists():
                present_suffixes.append(suffix)
                size = path.stat().st_size
                entry["sizeBytes"] = size
                entry["sizeHuman"] = file_size_human(size)

                if size == 0:
                    results["allJsonValid"] = False
                    entry["jsonValid"] = False
                    entry["error"] = "empty file"
                else:
                    valid, error = validate_json(path)
                    entry["jsonValid"] = valid
                    entry["error"] = error
                    if valid:
                        present_valid_suffixes.append(suffix)
                    else:
                        results["allJsonValid"] = False
            else:
                entry["jsonValid"] = False
                entry["error"] = "missing"

            group_entries.append(entry)

        if not present_valid_suffixes:
            results["allRequiredPresent"] = False
            if not present_suffixes:
                results["allJsonValid"] = False

        results["jsonOneOfChecks"].append(
            {
                "suffixes": group,
                "satisfied": bool(present_valid_suffixes),
                "presentSuffixes": present_suffixes,
                "presentValidSuffixes": present_valid_suffixes,
                "entries": group_entries,
            }
        )

    for suffix in optional_json_suffixes or []:
        path = base_dir / f"{prefix}_{suffix}.json"
        entry = {
            "suffix": suffix,
            "path": str(path),
            "exists": path.exists(),
            "sizeBytes": None,
            "sizeHuman": None,
            "jsonValid": None,
            "error": None,
        }

        if path.exists():
            size = path.stat().st_size
            entry["sizeBytes"] = size
            entry["sizeHuman"] = file_size_human(size)

            if size == 0:
                results["allJsonValid"] = False
                entry["jsonValid"] = False
                entry["error"] = "empty file"
            else:
                valid, error = validate_json(path)
                entry["jsonValid"] = valid
                entry["error"] = error
                if not valid:
                    results["allJsonValid"] = False

        results["optionalJsonChecks"].append(entry)

    for suffix in png_suffixes:
        path = base_dir / f"{prefix}_{suffix}.png"
        entry = {
            "suffix": suffix,
            "path": str(path),
            "exists": path.exists(),
            "sizeBytes": None,
            "sizeHuman": None,
            "error": None,
        }

        if not path.exists():
            results["allRequiredPresent"] = False
            entry["error"] = "missing"
        else:
            size = path.stat().st_size
            entry["sizeBytes"] = size
            entry["sizeHuman"] = file_size_human(size)
            if size == 0:
                entry["error"] = "empty file"

        results["pngChecks"].append(entry)

    json_png_files = sorted(base_dir.glob(f"{prefix}_*.json.png"))
    if json_png_files:
        results["hasJsonPngProblem"] = True
        results["unexpectedJsonPngFiles"] = [
            {
                "path": str(path),
                "sizeBytes": path.stat().st_size,
                "sizeHuman": file_size_human(path.stat().st_size),
            }
            for path in json_png_files
        ]

    return results


def build_overall_status(results: Dict[str, Any]) -> str:
    if results["hasJsonPngProblem"]:
        return "ERROR_JSON_PNG_FILES_FOUND"
    if not results["allRequiredPresent"]:
        return "ERROR_MISSING_REQUIRED_FILES"
    if not results["allJsonValid"]:
        return "ERROR_INVALID_JSON"
    return "OK"


def print_summary(results: Dict[str, Any]) -> None:
    status = build_overall_status(results)

    print("Step file check summary")
    print(f"- baseDir: {results['baseDir']}")
    print(f"- prefix: {results['prefix']}")
    print(f"- status: {status}")
    print(f"- allRequiredPresent: {results['allRequiredPresent']}")
    print(f"- allJsonValid: {results['allJsonValid']}")
    print(f"- hasJsonPngProblem: {results['hasJsonPngProblem']}")

    print("\nJSON files:")
    for entry in results["jsonChecks"]:
        mark = "OK" if entry["exists"] and entry["jsonValid"] else "NG"
        size = entry["sizeHuman"] if entry["sizeHuman"] else "-"
        error = f" error={entry['error']}" if entry["error"] else ""
        print(f"- [{mark}] {Path(entry['path']).name} size={size}{error}")

    if results["jsonOneOfChecks"]:
        print("\nJSON one-of required groups:")
        for group in results["jsonOneOfChecks"]:
            suffixes = ", ".join(group["suffixes"])
            present = ", ".join(group["presentValidSuffixes"]) or "-"
            mark = "OK" if group["satisfied"] else "NG"
            print(f"- [{mark}] one of: {suffixes} presentValid={present}")
            for entry in group["entries"]:
                mark = "OK" if entry["exists"] and entry["jsonValid"] else "NG"
                size = entry["sizeHuman"] if entry["sizeHuman"] else "-"
                error = f" error={entry['error']}" if entry["error"] else ""
                print(f"  - [{mark}] {Path(entry['path']).name} size={size}{error}")

    if results["optionalJsonChecks"]:
        print("\nOptional JSON files:")
        for entry in results["optionalJsonChecks"]:
            if not entry["exists"]:
                mark = "MISS"
            else:
                mark = "OK" if entry["jsonValid"] else "NG"
            size = entry["sizeHuman"] if entry["sizeHuman"] else "-"
            error = f" error={entry['error']}" if entry["error"] else ""
            print(f"- [{mark}] {Path(entry['path']).name} size={size}{error}")

    print("\nPNG files:")
    for entry in results["pngChecks"]:
        mark = "OK" if entry["exists"] and not entry["error"] else "NG"
        size = entry["sizeHuman"] if entry["sizeHuman"] else "-"
        error = f" error={entry['error']}" if entry["error"] else ""
        print(f"- [{mark}] {Path(entry['path']).name} size={size}{error}")

    if results["unexpectedJsonPngFiles"]:
        print("\nUnexpected .json.png files:")
        for entry in results["unexpectedJsonPngFiles"]:
            print(f"- {entry['path']} size={entry['sizeHuman']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check expected saved Step JSON/PNG files."
    )
    parser.add_argument(
        "--dir",
        required=True,
        help="Directory containing saved JSON/PNG files.",
    )
    parser.add_argument(
        "--prefix",
        required=True,
        help="Step file prefix, e.g. step107_000151_v13.",
    )
    parser.add_argument(
        "--json-suffixes",
        default=None,
        help=(
            "Comma-separated JSON suffix list. "
            "Default includes common Step103+ files."
        ),
    )
    parser.add_argument(
        "--png-suffixes",
        default=None,
        help="Comma-separated PNG suffix list. Default: canvas. Use an empty string for JSON-only captures.",
    )
    parser.add_argument(
        "--allow-missing-png",
        action="store_true",
        help="Do not require PNG files. Useful for compute-only or JSON-only captures.",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="Optional output path for check summary JSON.",
    )
    parser.add_argument(
        "--allow-missing-json-suffixes",
        default=None,
        help=(
            "Comma-separated JSON suffixes to remove from the required list. "
            "Useful when a Step does not produce every standard file."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    base_dir = Path(args.dir)
    prefix = args.prefix

    json_suffixes = (
        parse_csv_list(args.json_suffixes)
        if args.json_suffixes is not None
        else list(DEFAULT_JSON_SUFFIXES)
    )
    png_suffixes = (
        parse_csv_list(args.png_suffixes)
        if args.png_suffixes is not None
        else list(DEFAULT_PNG_SUFFIXES)
    )
    if args.allow_missing_png:
        png_suffixes = []

    allow_missing = set(parse_csv_list(args.allow_missing_json_suffixes))
    if allow_missing:
        json_suffixes = [suffix for suffix in json_suffixes if suffix not in allow_missing]
    json_one_of_groups = (
        []
        if args.json_suffixes is not None
        else [
            [suffix for suffix in group if suffix not in allow_missing]
            for group in DEFAULT_JSON_ONE_OF_GROUPS
        ]
    )
    json_one_of_groups = [group for group in json_one_of_groups if group]
    optional_json_suffixes = (
        []
        if args.json_suffixes is not None
        else [
            suffix
            for suffix in DEFAULT_OPTIONAL_JSON_SUFFIXES
            if suffix not in allow_missing
        ]
    )

    results = check_expected_files(
        base_dir=base_dir,
        prefix=prefix,
        json_suffixes=json_suffixes,
        png_suffixes=png_suffixes,
        json_one_of_groups=json_one_of_groups,
        optional_json_suffixes=optional_json_suffixes,
    )
    results["status"] = build_overall_status(results)
    results["requiredJsonSuffixes"] = json_suffixes
    results["requiredJsonOneOfGroups"] = json_one_of_groups
    results["optionalJsonSuffixes"] = optional_json_suffixes
    results["requiredPngSuffixes"] = png_suffixes

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print_summary(results)

    return 0 if results["status"] == "OK" else 2


if __name__ == "__main__":
    raise SystemExit(main())
