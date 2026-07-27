#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]


def generate(*args: str) -> dict[str, list[str]]:
    completed = subprocess.run(
        ["python3", "tools/make_step_url.py", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_qs(urlparse(completed.stdout.strip()).query)


def assert_production_webgpu(query: dict[str, list[str]]) -> None:
    assert query["viewerRuntime"] == ["webgpu"]
    assert query["gpuCandidateRuntime"] == ["cpu-reference"]
    assert query["webgpuBackendMode"] == ["webgpu-exclusive"]
    assert query["webgpuBackendImplementation"] == [
        "webgpu-tile-compositor-frame-implementation"
    ]
    assert query["webgpuAllowViewerCanvasPresentation"] == ["true"]
    assert query["webgpuBackendViewerLoopHook"] == ["true"]


def main() -> int:
    assert_production_webgpu(generate("--runtime", "webgpu"))
    assert_production_webgpu(
        generate("--preset", "stable", "--runtime", "webgpu")
    )
    print("step URL production runtime smoke tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
