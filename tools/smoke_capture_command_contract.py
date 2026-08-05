#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

from verify_capture_command_contract import build_capture_command_contract


ROOT = Path(__file__).resolve().parents[1]
POLICY = "fresh-production-diagnostic-json-png"


def generate_command(
    directory: Path,
    prefix: str,
    phase_step: str,
    comparison_mode: str,
) -> Path:
    output = directory / f"{prefix}_capture_commands.js"
    subprocess.run(
        [
            "python3",
            str(ROOT / "tools" / "make_capture_commands.py"),
            "--step",
            prefix,
            "--preset",
            "runtime-only",
            "--capture-lifecycle",
            POLICY,
            "--phase-step",
            phase_step,
            "--comparison-mode",
            comparison_mode,
            "--include-webgpu-visible-record-dryrun",
            "true",
            "--include-runtime",
            "true",
            "--include-png",
            "true",
            "--include-camera-control-debug",
            "false",
            "--webgpu-backend-implementation",
            "webgpu-tile-compositor-frame-implementation",
            "--expected-runtime",
            "webgpu",
            "--expected-effective-display-runtime",
            "webgpu-production",
            "--expected-webgpu-backend-mode",
            "webgpu-exclusive",
            "--expected-webgpu-backend-implementation",
            "webgpu-tile-compositor-frame-implementation",
            "--expected-webgpu-canvas-presentation",
            "true",
            "--expected-webgpu-viewer-loop-hook",
            "true",
            "--out",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(["node", "--check", str(output)], cwd=ROOT, check=True)
    return output


def write_contract(command_path: Path, output_path: Path) -> dict:
    contract = build_capture_command_contract(command_path)
    output_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return contract


def duplicate_fresh_request(source: str) -> str:
    match = re.search(
        r"genericFreshProductionRequest\s*=\s*await\s+"
        r"window\.gpuViewerDebug\.scheduleRender\s*\(\{.*?\}\);",
        source,
        re.DOTALL,
    )
    assert match
    return source[: match.end()] + "\n" + match.group(0) + source[match.end() :]


def move_diagnostic_before_completion(source: str) -> str:
    diagnostic = (
        "await window.gpuViewerDebug.captureWebGpuVisibleRecordDryRunDebug({});\n"
    )
    original = "await window.gpuViewerDebug.captureWebGpuVisibleRecordDryRunDebug({"
    assert source.count(original) == 1
    source = source.replace(original, "await Promise.resolve({", 1)
    marker = (
        "genericFreshProductionCaptureLifecycle.productionCompletionFence ="
    )
    index = source.index(marker)
    return source[:index] + diagnostic + source[index:]


def move_png_before_diagnostic(source: str) -> str:
    match = re.search(
        r"await\s+window\.gpuViewerDebug\.saveCurrentCanvasPng\s*"
        r"\(\{.*?\}\);",
        source,
        re.DOTALL,
    )
    assert match
    png_call = match.group(0)
    source = source[: match.start()] + source[match.end() :]
    diagnostic_marker = (
        "await window.gpuViewerDebug.captureWebGpuVisibleRecordDryRunDebug("
    )
    index = source.index(diagnostic_marker)
    return source[:index] + png_call + "\n" + source[index:]


def build_negative_sources(source: str) -> dict[str, str]:
    return {
        "duplicate-fresh-request": duplicate_fresh_request(source),
        "diagnostic-before-completion": move_diagnostic_before_completion(source),
        "png-before-diagnostic": move_png_before_diagnostic(source),
        "render-before-capture-true": source.replace(
            "renderBeforeCapture: false", "renderBeforeCapture: true", 1
        ),
        "schedule-after-png": source
        + "\nawait window.gpuViewerDebug.scheduleRender({"
        + " forceProductionUpdate: true });\n",
        "completion-fence-deleted": source.replace(
            "genericFreshProductionCaptureLifecycle.productionCompletionFence =",
            "genericRemovedProductionCompletionFence =",
            1,
        ),
    }


def run_summary(
    directory: Path,
    prefix: str,
    expected_returncode: int = 0,
) -> dict:
    output = directory / f"{prefix}_summary.json"
    completed = subprocess.run(
        [
            "python3",
            str(ROOT / "tools" / "summarize_step_json.py"),
            "--dir",
            str(directory),
            "--prefix",
            prefix,
            "--json",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
    )
    assert completed.returncode == expected_returncode
    return json.loads(output.read_text(encoding="utf-8"))


def main() -> int:
    with tempfile.TemporaryDirectory() as temp_dir:
        directory = Path(temp_dir)
        fixtures = {
            "phase3_step116_fix1_confirmation_a_000151_v13": (
                "phase3-step117",
                "phase3-step117-production-capture-diagnostic-isolation-recovery",
            ),
            "generic_capture_policy_fix1_confirmation": (
                "generic-phase-step",
                "generic-comparison-mode",
            ),
        }
        contracts = {}
        summaries = {}
        for prefix, (phase_step, comparison_mode) in fixtures.items():
            command_path = generate_command(
                directory, prefix, phase_step, comparison_mode
            )
            contract_path = directory / f"{prefix}_capture_command_contract.json"
            contract = write_contract(command_path, contract_path)
            assert contract["decision"] == "ready"
            assert contract["policy"] == POLICY
            assert contract["phaseStep"] == phase_step
            assert contract["comparisonMode"] == comparison_mode
            assert contract["counts"]["freshProductionRequest"] == 1
            assert contract["counts"]["forceProductionUpdateTrue"] == 1
            assert contract["predicates"]["completionFenceBeforeDiagnostic"] is True
            assert contract["predicates"]["pngIsLastArtifactSave"] is True
            assert contract["predicates"]["pngRenderBeforeCaptureFalse"] is True
            assert (
                contract["predicates"]["pngAfterProductionMutationAbsent"] is True
            )
            smoke_path = directory / f"{prefix}_capture_command_boundary_smoke.json"
            subprocess.run(
                [
                    "python3",
                    str(ROOT / "tools" / "smoke_capture_command_boundary.py"),
                    "--json",
                    str(smoke_path),
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            contracts[prefix] = contract
            summaries[prefix] = run_summary(directory, prefix)

        contract_a, contract_b = contracts.values()
        assert [stage["name"] for stage in contract_a["stages"]] == [
            stage["name"] for stage in contract_b["stages"]
        ]
        assert contract_a["counts"] == contract_b["counts"]
        assert contract_a["predicates"] == contract_b["predicates"]

        source = next(directory.glob("phase3_step116_fix1*_capture_commands.js"))
        source_text = source.read_text(encoding="utf-8")
        for name, negative_source in build_negative_sources(source_text).items():
            negative_path = directory / f"negative_{name}.js"
            negative_path.write_text(negative_source, encoding="utf-8")
            negative_contract = build_capture_command_contract(negative_path)
            assert negative_contract["decision"] == "blocked", name
            assert negative_contract["verificationErrors"], name

        for prefix, summary in summaries.items():
            assert summary["files"]
            assert summary["captureCommandContract"] is not None
            assert summary["captureCommandRegression"] is not None
            assert summary["captureCommandContract"]["decision"] == "ready"
            assert summary["captureCommandRegression"]["decision"] == "ready"
            assert summary["loadErrors"] == {}, prefix
            assert "candidate" in summary
            assert "runtime" in summary
            assert "step114CudaReferenceProvenance" in summary

        invalid_prefix = "invalid_capture_contract_schema"
        (directory / f"{invalid_prefix}_capture_command_contract.json").write_text(
            json.dumps({"schemaVersion": "invalid-schema"}),
            encoding="utf-8",
        )
        invalid_summary = run_summary(directory, invalid_prefix, expected_returncode=2)
        assert "capture_command_contract" in invalid_summary["loadErrors"]

    print("capture command contract smoke tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
