#!/usr/bin/env python3
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPORTER_PATH = REPOSITORY_ROOT / "converter" / "export_splat4d_from_ckpt.py"


def load_exporter():
    spec = importlib.util.spec_from_file_location("step119_exporter", EXPORTER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXPORTER = load_exporter()


def make_fields(record_count=5):
    sequence = np.arange(record_count, dtype=np.float32).reshape(record_count, 1)
    return {
        "xyz": sequence + np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
        "rotation": sequence + np.array(
            [[1.0, 0.25, 0.5, 0.75]], dtype=np.float32
        ),
        "rotation_r": sequence + np.array(
            [[0.75, 0.5, 0.25, 1.0]], dtype=np.float32
        ),
        "scale_xyz": sequence * 0.01 + np.array(
            [[-1.0, -0.5, -0.25]], dtype=np.float32
        ),
        "f_dc": np.stack(
            [sequence + 1.0, sequence + 2.0, sequence + 3.0], axis=2
        ),
        "f_rest": np.stack(
            [
                np.repeat(sequence + 4.0, 8, axis=1),
                np.repeat(sequence + 5.0, 8, axis=1),
                np.repeat(sequence + 6.0, 8, axis=1),
            ],
            axis=2,
        ),
        "opacity": sequence * 0.1 - 0.5,
        "t": sequence * 0.2 + 1.0,
        "scale_t": sequence * 0.03 - 0.75,
    }


def save_checkpoint(path, fields, iteration=12000):
    tensor = lambda value: torch.from_numpy(np.asarray(value, dtype=np.float32))
    record_count = fields["xyz"].shape[0]
    model_params = (
        2,
        tensor(fields["xyz"]),
        tensor(fields["f_dc"]),
        tensor(fields["f_rest"]),
        tensor(fields["scale_xyz"]),
        tensor(fields["rotation"]),
        tensor(fields["opacity"]),
        torch.zeros(record_count),
        torch.zeros(record_count, 1),
        torch.zeros(record_count, 1),
        torch.zeros(record_count, 1),
        None,
        1.0,
        tensor(fields["t"]),
        tensor(fields["scale_t"]),
        tensor(fields["rotation_r"]),
        True,
        None,
        1,
    )
    torch.save((model_params, iteration), path)


def old_v2_bytes(fields, store_scale_log=False):
    record_count = fields["xyz"].shape[0]
    flat = {
        name: np.asarray(fields[name], dtype=np.float32).reshape(record_count, -1)
        for name in EXPORTER.SPL4_V2_FIELD_ORDER
    }
    if not store_scale_log:
        flat["scale_xyz"] = np.exp(flat["scale_xyz"]).astype(np.float32)
        flat["scale_t"] = np.exp(flat["scale_t"]).astype(np.float32)
    dimensions = [flat[name].shape[1] for name in EXPORTER.SPL4_V2_FIELD_ORDER]
    header = struct.pack(
        "<4sIIiiBBBB10I",
        b"SPL4",
        2,
        record_count,
        2,
        1,
        1,
        1 if store_scale_log else 0,
        1,
        1,
        *dimensions,
        0,
    )
    result = bytearray(header + b"\x00" * (128 - len(header)))
    for index in range(record_count):
        row = np.concatenate(
            [flat[name][index] for name in EXPORTER.SPL4_V2_FIELD_ORDER]
        ).astype(np.float32)
        result.extend(row.tobytes(order="C"))
    return bytes(result)


def export_asset(path, fields, store_scale_log=False):
    EXPORTER.export_v2(
        out_path=path,
        active_sh_degree=2,
        active_sh_degree_t=1,
        rot_4d=True,
        store_scale_log=store_scale_log,
        xyz=fields["xyz"],
        rotation=fields["rotation"],
        rotation_r=fields["rotation_r"],
        scaling_xyz=fields["scale_xyz"],
        f_dc=fields["f_dc"],
        f_rest=fields["f_rest"],
        opacity=fields["opacity"],
        t=fields["t"],
        scaling_t=fields["scale_t"],
    )


def verify(checkpoint, asset, start, count):
    return EXPORTER.verify_checkpoint_spl4_v2_population_range(
        checkpoint_path=checkpoint,
        asset_path=asset,
        start_inclusive=start,
        selected_record_count=count,
    )


def replace_bytes(source, destination, offset, replacement):
    data = bytearray(Path(source).read_bytes())
    data[offset:offset + len(replacement)] = replacement
    Path(destination).write_bytes(data)


def assert_ready(result):
    assert result["decision"] == "ready", result
    assert result["blockedReasons"] == [], result
    assert result["rangeHashesMatch"] is True, result
    assert result["exactPopulationMappingReady"] is True, result


def assert_blocked(result, reason):
    assert result["decision"] == "blocked", result
    assert result["exactPopulationMappingReady"] is False, result
    assert reason in result["blockedReasons"], result


def main():
    fields = make_fields()
    with tempfile.TemporaryDirectory(prefix="step119-provenance-") as temp_dir:
        temp = Path(temp_dir)
        checkpoint = temp / "fixture.pth"
        asset = temp / "fixture.splat4d"
        save_checkpoint(checkpoint, fields)
        export_asset(asset, fields)

        # The shared serializer preserves the pre-Fix1 v2 binary byte-for-byte.
        expected_asset = old_v2_bytes(fields, store_scale_log=False)
        assert asset.read_bytes() == expected_asset

        full_result = verify(checkpoint, asset, 0, len(fields["xyz"]))
        assert_ready(full_result)
        assert full_result["expectedSerializedByteCount"] == len(expected_asset) - 128
        assert full_result["actualAssetRangeByteCount"] == len(expected_asset) - 128

        subset_result = verify(checkpoint, asset, 1, 3)
        assert_ready(subset_result)
        assert subset_result["selection"] == {
            "policy": "contiguous-source-index-range",
            "startInclusive": 1,
            "endExclusive": 4,
            "selectedRecordCount": 3,
        }

        asset_hash_before = hashlib.sha256(asset.read_bytes()).hexdigest()
        verify(checkpoint, asset, 0, 2)
        assert hashlib.sha256(asset.read_bytes()).hexdigest() == asset_hash_before

        mutated_asset = temp / "mutated-payload.splat4d"
        original = bytearray(asset.read_bytes())
        original[128 + full_result["spl4Format"]["recordStrideBytes"] + 3] ^= 1
        mutated_asset.write_bytes(original)
        assert_blocked(
            verify(checkpoint, mutated_asset, 1, 1),
            "checkpoint-asset-range-hash-mismatch",
        )

        mutated_fields = copy.deepcopy(fields)
        mutated_fields["opacity"] = mutated_fields["opacity"].copy()
        mutated_fields["opacity"][2, 0] += np.float32(0.125)
        mutated_checkpoint = temp / "mutated-checkpoint.pth"
        save_checkpoint(mutated_checkpoint, mutated_fields)
        assert_blocked(
            verify(mutated_checkpoint, asset, 2, 1),
            "checkpoint-asset-range-hash-mismatch",
        )

        count_mismatch_asset = temp / "count-mismatch.splat4d"
        replace_bytes(
            asset,
            count_mismatch_asset,
            8,
            struct.pack("<I", len(fields["xyz"]) - 1),
        )
        assert_blocked(
            verify(checkpoint, count_mismatch_asset, 0, 1),
            "checkpoint-asset-record-count-mismatch",
        )

        assert_blocked(
            verify(checkpoint, asset, len(fields["xyz"]) - 1, 2),
            "selection-range-out-of-bounds",
        )

        dimension_mismatch_asset = temp / "dimension-mismatch.splat4d"
        replace_bytes(asset, dimension_mismatch_asset, 24, struct.pack("<I", 4))
        assert_blocked(
            verify(checkpoint, dimension_mismatch_asset, 0, 1),
            "checkpoint-spl4-format-mismatch",
        )

        scale_policy_asset = temp / "scale-policy-mismatch.splat4d"
        replace_bytes(asset, scale_policy_asset, 21, b"\x01")
        assert_blocked(
            verify(checkpoint, scale_policy_asset, 0, len(fields["xyz"])),
            "checkpoint-asset-range-hash-mismatch",
        )

        # Exercise the public CLI and prove stdout equals the optional JSON file.
        result_json = temp / "result.json"
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(EXPORTER_PATH),
                "--ckpt",
                str(checkpoint),
                "--verify-existing-v2",
                str(asset),
                "--verify-start",
                "1",
                "--verify-count",
                "3",
                "--verification-json",
                str(result_json),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert json.loads(completed.stdout) == json.loads(result_json.read_text())
        assert_ready(json.loads(completed.stdout))

        # Exercise the unchanged normal CLI default (v2, activated scales).
        cli_asset = temp / "normal-cli.splat4d"
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(EXPORTER_PATH),
                "--ckpt",
                str(checkpoint),
                "--out",
                str(cli_asset),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert cli_asset.read_bytes() == expected_asset

    print("step119 population provenance smoke: ok")


if __name__ == "__main__":
    main()
