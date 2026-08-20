import struct
import hashlib
import json
import os
import numpy as np
import torch
import argparse


SPL4_V2_HEADER_SIZE = 128
SPL4_V2_VERSION = 2
SPL4_V2_BYTE_ORDER = "little-endian"
SPL4_V2_FIELD_ORDER = (
    "xyz",
    "rotation",
    "rotation_r",
    "scale_xyz",
    "f_dc",
    "f_rest",
    "opacity",
    "t",
    "scale_t",
)
SPL4_V2_SCALE_FIELDS = frozenset(("scale_xyz", "scale_t"))
SPL4_V2_SERIALIZATION_CHUNK_RECORDS = 8192
POPULATION_PROVENANCE_SCHEMA_VERSION = (
    "phase3-checkpoint-spl4-population-provenance-v1"
)
POPULATION_PROVENANCE_VERIFICATION_MODE = (
    "checkpoint-existing-spl4-v2-contiguous-source-index-range"
)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def to_numpy_f32(x, name):
    if x is None:
        raise ValueError(f"{name} is None")
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be a torch.Tensor")
    return x.detach().cpu().numpy().astype(np.float32)


def to_per_gauss_2d(arr, name, N):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape[0] != N:
        raise ValueError(f"{name}: first dimension mismatch: {arr.shape[0]} != {N}")
    return arr.reshape(N, -1).astype(np.float32)


def to_scalar_int(x, name):
    if torch.is_tensor(x):
        return int(x.detach().cpu().item())
    return int(x)


def to_scalar_bool(x, name):
    if torch.is_tensor(x):
        return bool(x.detach().cpu().item())
    return bool(x)


def sha256_file(path, chunk_bytes=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_file_identity(path):
    absolute_path = os.path.abspath(path)
    return {
        "absolutePath": absolute_path,
        "sizeBytes": os.path.getsize(absolute_path),
        "sha256": sha256_file(absolute_path),
    }


def unpack_checkpoint_capture(checkpoint):
    if not (isinstance(checkpoint, (tuple, list)) and len(checkpoint) == 2):
        raise TypeError("Expected checkpoint format: (model_params, iteration)")
    model_params, iteration = checkpoint
    if not (isinstance(model_params, (tuple, list)) and len(model_params) == 19):
        raise TypeError("Expected model_params capture() format len == 19")

    (
        active_sh_degree,
        xyz, f_dc, f_rest,
        scaling_xyz, rotation, opacity,
        max_radii2D, xyz_grad_accum, t_grad_accum, denom,
        opt_state, spatial_lr_scale,
        t, scaling_t, rotation_r, rot_4d, env_map, active_sh_degree_t,
    ) = model_params
    del max_radii2D, xyz_grad_accum, t_grad_accum, denom
    del opt_state, spatial_lr_scale, env_map

    fields = {
        "xyz": xyz,
        "rotation": rotation,
        "rotation_r": rotation_r,
        "scale_xyz": scaling_xyz,
        "f_dc": f_dc,
        "f_rest": f_rest,
        "opacity": opacity,
        "t": t,
        "scale_t": scaling_t,
    }
    return {
        "iteration": to_scalar_int(iteration, "iteration"),
        "activeShDegree": to_scalar_int(active_sh_degree, "active_sh_degree"),
        "activeShDegreeT": to_scalar_int(
            active_sh_degree_t, "active_sh_degree_t"
        ),
        "rot4d": to_scalar_bool(rot_4d, "rot_4d"),
        "fields": fields,
    }


def per_gaussian_tensor_dimension(value, name, record_count):
    if value is None:
        raise ValueError(f"{name} is None")
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    shape = tuple(value.shape)
    if not shape or int(shape[0]) != record_count:
        first_dimension = shape[0] if shape else None
        raise ValueError(
            f"{name}: first dimension mismatch: {first_dimension} != {record_count}"
        )
    dimension = int(np.prod(shape[1:], dtype=np.int64))
    if dimension <= 0:
        raise ValueError(f"{name}: per-record dimension must be positive")
    return dimension


def prepare_v2_field_arrays(fields, store_scale_log, record_count=None):
    if record_count is None:
        record_count = int(fields["xyz"].shape[0])
    prepared = {}
    for name in SPL4_V2_FIELD_ORDER:
        value = fields[name]
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        array = to_per_gauss_2d(value, name, record_count)
        if name in SPL4_V2_SCALE_FIELDS and not store_scale_log:
            array = np.exp(array).astype(np.float32)
        prepared[name] = array
    return prepared


def v2_field_dimensions(fields):
    return {name: int(fields[name].shape[1]) for name in SPL4_V2_FIELD_ORDER}


def serialize_v2_record_chunk(fields):
    if not fields:
        return b""
    row_count = int(fields[SPL4_V2_FIELD_ORDER[0]].shape[0])
    for name in SPL4_V2_FIELD_ORDER:
        if int(fields[name].shape[0]) != row_count:
            raise ValueError(f"{name}: chunk record count mismatch")
    if row_count == 0:
        return b""
    records = np.concatenate(
        [fields[name] for name in SPL4_V2_FIELD_ORDER], axis=1
    ).astype("<f4", copy=False)
    return records.tobytes(order="C")


def iter_v2_serialized_range(
    fields,
    start_inclusive,
    end_exclusive,
    store_scale_log,
    chunk_records=SPL4_V2_SERIALIZATION_CHUNK_RECORDS,
):
    if chunk_records <= 0:
        raise ValueError("chunk_records must be positive")
    for chunk_start in range(start_inclusive, end_exclusive, chunk_records):
        chunk_end = min(end_exclusive, chunk_start + chunk_records)
        sliced = {
            name: fields[name][chunk_start:chunk_end]
            for name in SPL4_V2_FIELD_ORDER
        }
        prepared = prepare_v2_field_arrays(
            sliced,
            store_scale_log=store_scale_log,
            record_count=chunk_end - chunk_start,
        )
        yield serialize_v2_record_chunk(prepared)


def parse_spl4_v2_header(path):
    with open(path, "rb") as f:
        header_bytes = f.read(SPL4_V2_HEADER_SIZE)
    if len(header_bytes) != SPL4_V2_HEADER_SIZE:
        raise ValueError(
            f"SPL4 header is truncated: {len(header_bytes)} != {SPL4_V2_HEADER_SIZE}"
        )

    unpacked = struct.unpack_from("<4sIIiiBBBB10I", header_bytes, 0)
    (
        magic_bytes,
        version,
        record_count,
        active_sh_degree,
        active_sh_degree_t,
        rot_4d_raw,
        store_scale_log_raw,
        raw_sh,
        raw_opacity,
        xyz_dim,
        rotation_dim,
        rotation_r_dim,
        scale_xyz_dim,
        f_dc_dim,
        f_rest_dim,
        opacity_dim,
        t_dim,
        scale_t_dim,
        reserved_0,
    ) = unpacked
    field_dimensions = {
        "xyz": xyz_dim,
        "rotation": rotation_dim,
        "rotation_r": rotation_r_dim,
        "scale_xyz": scale_xyz_dim,
        "f_dc": f_dc_dim,
        "f_rest": f_rest_dim,
        "opacity": opacity_dim,
        "t": t_dim,
        "scale_t": scale_t_dim,
    }
    return {
        "magic": magic_bytes.decode("ascii", errors="replace"),
        "version": version,
        "headerSize": SPL4_V2_HEADER_SIZE,
        "recordStrideBytes": sum(field_dimensions.values()) * 4,
        "recordCount": record_count,
        "activeShDegree": active_sh_degree,
        "activeShDegreeT": active_sh_degree_t,
        "rot4d": bool(rot_4d_raw),
        "storeScaleLog": bool(store_scale_log_raw),
        "rawSh": bool(raw_sh),
        "rawOpacity": bool(raw_opacity),
        "fieldDimensions": field_dimensions,
        "byteOrder": SPL4_V2_BYTE_ORDER,
        "contractBytes": {
            "rot4d": rot_4d_raw,
            "storeScaleLog": store_scale_log_raw,
            "rawSh": raw_sh,
            "rawOpacity": raw_opacity,
            "reserved0": reserved_0,
            "reservedPaddingAllZero": all(
                byte == 0 for byte in header_bytes[64:SPL4_V2_HEADER_SIZE]
            ),
        },
    }


def build_blocked_provenance_result(checkpoint_path, asset_path, start, count):
    return {
        "schemaVersion": POPULATION_PROVENANCE_SCHEMA_VERSION,
        "decision": "blocked",
        "blockedReasons": [],
        "verificationMode": POPULATION_PROVENANCE_VERIFICATION_MODE,
        "checkpoint": {
            "absolutePath": os.path.abspath(checkpoint_path),
            "sizeBytes": None,
            "sha256": None,
        },
        "spl4Asset": {
            "absolutePath": os.path.abspath(asset_path),
            "sizeBytes": None,
            "sha256": None,
        },
        "spl4Format": None,
        "checkpointFormat": None,
        "checkpointSourceGaussianCount": None,
        "assetRecordCount": None,
        "selection": {
            "policy": "contiguous-source-index-range",
            "startInclusive": start,
            "endExclusive": start + max(0, count),
            "selectedRecordCount": count,
        },
        "expectedSerializedByteCount": None,
        "actualAssetRangeByteCount": None,
        "checkpointRangeSha256": None,
        "assetRangeSha256": None,
        "rangeHashesMatch": False,
        "headerCompatible": False,
        "recordCountsMatch": False,
        "assetPayloadLengthValid": False,
        "rangeInBounds": False,
        "exactPopulationMappingReady": False,
    }


def append_blocked_reason(result, reason):
    if reason not in result["blockedReasons"]:
        result["blockedReasons"].append(reason)


def hash_checkpoint_serialized_range(
    fields, start, end, store_scale_log, expected_byte_count
):
    digest = hashlib.sha256()
    byte_count = 0
    for chunk in iter_v2_serialized_range(
        fields,
        start,
        end,
        store_scale_log=store_scale_log,
    ):
        digest.update(chunk)
        byte_count += len(chunk)
    if byte_count != expected_byte_count:
        raise ValueError(
            f"checkpoint serialized byte count mismatch: {byte_count} != "
            f"{expected_byte_count}"
        )
    return digest.hexdigest(), byte_count


def hash_asset_range(path, byte_offset, expected_byte_count, chunk_bytes=1024 * 1024):
    digest = hashlib.sha256()
    byte_count = 0
    with open(path, "rb") as f:
        f.seek(byte_offset)
        remaining = expected_byte_count
        while remaining:
            chunk = f.read(min(remaining, chunk_bytes))
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
            remaining -= len(chunk)
    return digest.hexdigest(), byte_count


def verify_checkpoint_spl4_v2_population_range(
    checkpoint_path,
    asset_path,
    start_inclusive,
    selected_record_count,
):
    result = build_blocked_provenance_result(
        checkpoint_path,
        asset_path,
        start_inclusive,
        selected_record_count,
    )
    try:
        result["checkpoint"] = build_file_identity(checkpoint_path)
        result["spl4Asset"] = build_file_identity(asset_path)
        header = parse_spl4_v2_header(asset_path)
        result["spl4Format"] = header
        result["assetRecordCount"] = header["recordCount"]

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        capture = unpack_checkpoint_capture(checkpoint)
        del checkpoint
        fields = capture["fields"]
        source_count = int(fields["xyz"].shape[0])
        result["checkpointSourceGaussianCount"] = source_count

        checkpoint_dimensions = {
            name: per_gaussian_tensor_dimension(fields[name], name, source_count)
            for name in SPL4_V2_FIELD_ORDER
        }
        result["checkpointFormat"] = {
            "iteration": capture["iteration"],
            "activeShDegree": capture["activeShDegree"],
            "activeShDegreeT": capture["activeShDegreeT"],
            "rot4d": capture["rot4d"],
            "fieldDimensions": checkpoint_dimensions,
        }

        contract_bytes = header["contractBytes"]
        header_primitives_valid = (
            header["magic"] == "SPL4"
            and header["version"] == SPL4_V2_VERSION
            and contract_bytes["rot4d"] in (0, 1)
            and contract_bytes["storeScaleLog"] in (0, 1)
            and contract_bytes["rawSh"] == 1
            and contract_bytes["rawOpacity"] == 1
            and contract_bytes["reserved0"] == 0
            and contract_bytes["reservedPaddingAllZero"]
            and all(value > 0 for value in header["fieldDimensions"].values())
        )
        semantic_header_match = (
            header["activeShDegree"] == capture["activeShDegree"]
            and header["activeShDegreeT"] == capture["activeShDegreeT"]
            and header["rot4d"] == capture["rot4d"]
            and header["fieldDimensions"] == checkpoint_dimensions
        )
        result["headerCompatible"] = bool(
            header_primitives_valid and semantic_header_match
        )
        if not header_primitives_valid:
            append_blocked_reason(result, "spl4-v2-header-contract-invalid")
        if not semantic_header_match:
            append_blocked_reason(result, "checkpoint-spl4-format-mismatch")

        result["recordCountsMatch"] = source_count == header["recordCount"]
        if not result["recordCountsMatch"]:
            append_blocked_reason(result, "checkpoint-asset-record-count-mismatch")

        expected_asset_size = (
            SPL4_V2_HEADER_SIZE
            + header["recordCount"] * header["recordStrideBytes"]
        )
        payload_length_valid = result["spl4Asset"]["sizeBytes"] == expected_asset_size
        result["assetPayloadLengthValid"] = payload_length_valid
        if not payload_length_valid:
            append_blocked_reason(result, "spl4-payload-length-mismatch")

        end_exclusive = start_inclusive + max(0, selected_record_count)
        range_valid = (
            start_inclusive >= 0
            and selected_record_count > 0
            and end_exclusive <= source_count
            and end_exclusive <= header["recordCount"]
        )
        result["rangeInBounds"] = range_valid
        if not range_valid:
            append_blocked_reason(result, "selection-range-out-of-bounds")

        checkpoint_stride = sum(checkpoint_dimensions.values()) * 4
        expected_byte_count = max(0, selected_record_count) * checkpoint_stride
        result["expectedSerializedByteCount"] = expected_byte_count
        if result["headerCompatible"] and range_valid:
            checkpoint_hash, checkpoint_bytes = hash_checkpoint_serialized_range(
                fields,
                start_inclusive,
                end_exclusive,
                store_scale_log=header["storeScaleLog"],
                expected_byte_count=expected_byte_count,
            )
            asset_hash, asset_bytes = hash_asset_range(
                asset_path,
                SPL4_V2_HEADER_SIZE
                + start_inclusive * header["recordStrideBytes"],
                expected_byte_count,
            )
            result["checkpointRangeSha256"] = checkpoint_hash
            result["assetRangeSha256"] = asset_hash
            result["actualAssetRangeByteCount"] = asset_bytes
            result["rangeHashesMatch"] = (
                checkpoint_bytes == expected_byte_count
                and asset_bytes == expected_byte_count
                and checkpoint_hash == asset_hash
            )
            if asset_bytes != expected_byte_count:
                append_blocked_reason(result, "asset-range-byte-count-mismatch")
            if not result["rangeHashesMatch"]:
                append_blocked_reason(result, "checkpoint-asset-range-hash-mismatch")

        result["exactPopulationMappingReady"] = bool(
            result["headerCompatible"]
            and result["recordCountsMatch"]
            and payload_length_valid
            and result["rangeInBounds"]
            and result["rangeHashesMatch"]
            and not result["blockedReasons"]
        )
        result["decision"] = (
            "ready" if result["exactPopulationMappingReady"] else "blocked"
        )
    except Exception as exc:
        append_blocked_reason(
            result,
            f"verification-error:{type(exc).__name__}:{exc}",
        )
    return result


def write_header_v2(
    f,
    N,
    active_sh_degree,
    active_sh_degree_t,
    rot_4d,
    store_scale_log,
    xyz_dim,
    rotation_dim,
    rotation_r_dim,
    scale_xyz_dim,
    f_dc_dim,
    f_rest_dim,
    opacity_dim,
    t_dim,
    scale_t_dim,
):
    # v2 header layout (128 bytes total)
    # 0   : 4s   magic = b"SPL4"
    # 4   : u32  version = 2
    # 8   : u32  N
    # 12  : i32  active_sh_degree
    # 16  : i32  active_sh_degree_t
    # 20  : u8   rot_4d
    # 21  : u8   store_scale_log
    # 22  : u8   raw_sh = 1
    # 23  : u8   raw_opacity = 1
    # 24  : u32  xyz_dim
    # 28  : u32  rotation_dim
    # 32  : u32  rotation_r_dim
    # 36  : u32  scale_xyz_dim
    # 40  : u32  f_dc_dim
    # 44  : u32  f_rest_dim
    # 48  : u32  opacity_dim
    # 52  : u32  t_dim
    # 56  : u32  scale_t_dim
    # 60  : u32  reserved_0
    # 64..127 reserved/padding

    f.write(b"SPL4")
    f.write(struct.pack("<I", 2))
    f.write(struct.pack("<I", N))
    f.write(struct.pack("<i", active_sh_degree))
    f.write(struct.pack("<i", active_sh_degree_t))
    f.write(struct.pack("<B", 1 if rot_4d else 0))
    f.write(struct.pack("<B", 1 if store_scale_log else 0))
    f.write(struct.pack("<B", 1))  # raw_sh
    f.write(struct.pack("<B", 1))  # raw_opacity
    f.write(struct.pack("<I", xyz_dim))
    f.write(struct.pack("<I", rotation_dim))
    f.write(struct.pack("<I", rotation_r_dim))
    f.write(struct.pack("<I", scale_xyz_dim))
    f.write(struct.pack("<I", f_dc_dim))
    f.write(struct.pack("<I", f_rest_dim))
    f.write(struct.pack("<I", opacity_dim))
    f.write(struct.pack("<I", t_dim))
    f.write(struct.pack("<I", scale_t_dim))
    f.write(struct.pack("<I", 0))  # reserved_0

    if f.tell() > 128:
      raise RuntimeError(f"Header size exceeded 128 bytes: {f.tell()}")
    f.write(b"\x00" * (128 - f.tell()))


def export_legacy_v1(
    out_path,
    active_sh_degree,
    active_sh_degree_t,
    rot_4d,
    store_scale_log,
    xyz,
    rotation,
    scaling_xyz,
    f_dc,
    opacity,
    t,
    scaling_t,
):
    # 旧viewer互換:
    # xyz(3), rot(4), scale_xyz(3), rgb(3), alpha(1), t(1), sigma_t(1)
    N = xyz.shape[0]

    rgb = f_dc[:, 0, :].astype(np.float32)
    rgb = np.clip(rgb + 0.5, 0.0, 1.0)

    alpha = sigmoid(opacity).astype(np.float32)

    sca = scaling_xyz.copy()
    st = scaling_t.copy()
    if not store_scale_log:
        sca = np.exp(sca).astype(np.float32)
        st = np.exp(st).astype(np.float32)

    with open(out_path, "wb") as f:
        f.write(b"SPL4")
        f.write(struct.pack("<I", N))
        f.write(struct.pack("<i", active_sh_degree))
        f.write(struct.pack("<i", active_sh_degree_t))
        f.write(struct.pack("<?", bool(rot_4d)))
        f.write(struct.pack("<?", bool(store_scale_log)))
        f.write(b"\x00" * (32 - f.tell()))

        for i in range(N):
            f.write(struct.pack(
                "<16f",
                xyz[i, 0], xyz[i, 1], xyz[i, 2],
                rotation[i, 0], rotation[i, 1], rotation[i, 2], rotation[i, 3],
                sca[i, 0], sca[i, 1], sca[i, 2],
                rgb[i, 0], rgb[i, 1], rgb[i, 2],
                float(alpha[i, 0]),
                float(t[i, 0]),
                float(st[i, 0]),
            ))


def export_v2(
    out_path,
    active_sh_degree,
    active_sh_degree_t,
    rot_4d,
    store_scale_log,
    xyz,
    rotation,
    rotation_r,
    scaling_xyz,
    f_dc,
    f_rest,
    opacity,
    t,
    scaling_t,
):
    N = xyz.shape[0]
    fields = prepare_v2_field_arrays(
        {
            "xyz": xyz,
            "rotation": rotation,
            "rotation_r": rotation_r,
            "scale_xyz": scaling_xyz,
            "f_dc": f_dc,
            "f_rest": f_rest,
            "opacity": opacity,
            "t": t,
            "scale_t": scaling_t,
        },
        store_scale_log=store_scale_log,
        record_count=N,
    )
    dimensions = v2_field_dimensions(fields)

    with open(out_path, "wb") as f:
        write_header_v2(
            f=f,
            N=N,
            active_sh_degree=active_sh_degree,
            active_sh_degree_t=active_sh_degree_t,
            rot_4d=rot_4d,
            store_scale_log=store_scale_log,
            xyz_dim=dimensions["xyz"],
            rotation_dim=dimensions["rotation"],
            rotation_r_dim=dimensions["rotation_r"],
            scale_xyz_dim=dimensions["scale_xyz"],
            f_dc_dim=dimensions["f_dc"],
            f_rest_dim=dimensions["f_rest"],
            opacity_dim=dimensions["opacity"],
            t_dim=dimensions["t"],
            scale_t_dim=dimensions["scale_t"],
        )
        for chunk_start in range(0, N, SPL4_V2_SERIALIZATION_CHUNK_RECORDS):
            chunk_end = min(N, chunk_start + SPL4_V2_SERIALIZATION_CHUNK_RECORDS)
            f.write(serialize_v2_record_chunk({
                name: fields[name][chunk_start:chunk_end]
                for name in SPL4_V2_FIELD_ORDER
            }))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="chkpnt_best.pth")
    ap.add_argument("--out", help="output .splat4d (export mode only)")
    ap.add_argument(
        "--store_scale_log",
        action="store_true",
        help="store xyz scale and sigma_t as log values (recommended)",
    )
    ap.add_argument(
        "--legacy_v1",
        action="store_true",
        help="write old 16-float format for old viewers",
    )
    ap.add_argument(
        "--verify-existing-v2",
        metavar="ASSET",
        help="verify checkpoint serialization against an existing SPL4-v2 asset",
    )
    ap.add_argument(
        "--verify-start",
        type=int,
        default=0,
        help="verification source-index range start (inclusive)",
    )
    ap.add_argument(
        "--verify-count",
        type=int,
        help="number of contiguous records to verify",
    )
    ap.add_argument(
        "--verification-json",
        metavar="PATH",
        help="also write the complete verification result JSON to PATH",
    )
    args = ap.parse_args()

    verification_mode = args.verify_existing_v2 is not None
    if verification_mode:
        if args.out is not None:
            ap.error("--out cannot be used with --verify-existing-v2")
        if args.legacy_v1 or args.store_scale_log:
            ap.error(
                "export policy flags cannot be used with --verify-existing-v2; "
                "verification uses the existing asset header policy"
            )
        if args.verify_count is None or args.verify_count <= 0:
            ap.error("--verify-count must be a positive integer")
        if args.verify_start < 0:
            ap.error("--verify-start must be non-negative")
        result = verify_checkpoint_spl4_v2_population_range(
            checkpoint_path=args.ckpt,
            asset_path=args.verify_existing_v2,
            start_inclusive=args.verify_start,
            selected_record_count=args.verify_count,
        )
        rendered = json.dumps(result, indent=2, sort_keys=True)
        if args.verification_json:
            output_path = os.path.abspath(args.verification_json)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(rendered)
                f.write("\n")
        print(rendered)
        raise SystemExit(0 if result["decision"] == "ready" else 1)

    if args.out is None:
        ap.error("--out is required in export mode")
    if args.verify_count is not None or args.verification_json is not None:
        ap.error(
            "--verify-count and --verification-json require --verify-existing-v2"
        )

    x = torch.load(args.ckpt, map_location="cpu")
    capture = unpack_checkpoint_capture(x)
    it = capture["iteration"]
    active_sh_degree = capture["activeShDegree"]
    active_sh_degree_t = capture["activeShDegreeT"]
    rot_4d = capture["rot4d"]
    fields = capture["fields"]

    xyz = fields["xyz"]
    f_dc = fields["f_dc"]
    f_rest = fields["f_rest"]
    scaling_xyz = fields["scale_xyz"]
    rotation = fields["rotation"]
    opacity = fields["opacity"]
    t = fields["t"]
    scaling_t = fields["scale_t"]
    rotation_r = fields["rotation_r"]

    xyz = to_numpy_f32(xyz, "xyz")
    f_dc = to_numpy_f32(f_dc, "f_dc")
    f_rest = to_numpy_f32(f_rest, "f_rest")
    scaling_xyz = to_numpy_f32(scaling_xyz, "scaling_xyz")
    rotation = to_numpy_f32(rotation, "rotation")
    opacity = to_numpy_f32(opacity, "opacity")
    t = to_numpy_f32(t, "t")
    scaling_t = to_numpy_f32(scaling_t, "scaling_t")
    rotation_r = to_numpy_f32(rotation_r, "rotation_r")

    N = xyz.shape[0]

    if args.legacy_v1:
        export_legacy_v1(
            out_path=args.out,
            active_sh_degree=active_sh_degree,
            active_sh_degree_t=active_sh_degree_t,
            rot_4d=rot_4d,
            store_scale_log=args.store_scale_log,
            xyz=to_per_gauss_2d(xyz, "xyz", N),
            rotation=to_per_gauss_2d(rotation, "rotation", N),
            scaling_xyz=to_per_gauss_2d(scaling_xyz, "scaling_xyz", N),
            f_dc=f_dc,
            opacity=to_per_gauss_2d(opacity, "opacity", N),
            t=to_per_gauss_2d(t, "t", N),
            scaling_t=to_per_gauss_2d(scaling_t, "scaling_t", N),
        )
        print("[DONE legacy_v1]", args.out)
    else:
        export_v2(
            out_path=args.out,
            active_sh_degree=active_sh_degree,
            active_sh_degree_t=active_sh_degree_t,
            rot_4d=rot_4d,
            store_scale_log=args.store_scale_log,
            xyz=xyz,
            rotation=rotation,
            rotation_r=rotation_r,
            scaling_xyz=scaling_xyz,
            f_dc=f_dc,
            f_rest=f_rest,
            opacity=opacity,
            t=t,
            scaling_t=scaling_t,
        )
        print("[DONE v2]", args.out)

    print("N =", N)
    print("iteration =", int(it))
    print("active_sh_degree =", active_sh_degree)
    print("active_sh_degree_t =", active_sh_degree_t)
    print("rot_4d =", rot_4d)
    print("store_scale_log =", bool(args.store_scale_log))

    print("per-gauss dims:")
    print("  xyz        =", to_per_gauss_2d(xyz, "xyz", N).shape[1])
    print("  rotation   =", to_per_gauss_2d(rotation, "rotation", N).shape[1])
    print("  rotation_r =", to_per_gauss_2d(rotation_r, "rotation_r", N).shape[1])
    print("  scale_xyz  =", to_per_gauss_2d(scaling_xyz, "scaling_xyz", N).shape[1])
    print("  f_dc       =", to_per_gauss_2d(f_dc, "f_dc", N).shape[1])
    print("  f_rest     =", to_per_gauss_2d(f_rest, "f_rest", N).shape[1])
    print("  opacity    =", to_per_gauss_2d(opacity, "opacity", N).shape[1])
    print("  t          =", to_per_gauss_2d(t, "t", N).shape[1])
    print("  scaling_t  =", to_per_gauss_2d(scaling_t, "scaling_t", N).shape[1])


if __name__ == "__main__":
    main()
