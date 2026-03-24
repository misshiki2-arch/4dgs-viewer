import struct
import numpy as np
import torch
import argparse


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

    xyz2 = to_per_gauss_2d(xyz, "xyz", N)
    rot2 = to_per_gauss_2d(rotation, "rotation", N)
    rotr2 = to_per_gauss_2d(rotation_r, "rotation_r", N)
    sca2 = to_per_gauss_2d(scaling_xyz, "scaling_xyz", N)
    fdc2 = to_per_gauss_2d(f_dc, "f_dc", N)
    frs2 = to_per_gauss_2d(f_rest, "f_rest", N)
    opa2 = to_per_gauss_2d(opacity, "opacity", N)
    t2 = to_per_gauss_2d(t, "t", N)
    st2 = to_per_gauss_2d(scaling_t, "scaling_t", N)

    if not store_scale_log:
        sca2 = np.exp(sca2).astype(np.float32)
        st2 = np.exp(st2).astype(np.float32)

    xyz_dim = xyz2.shape[1]
    rotation_dim = rot2.shape[1]
    rotation_r_dim = rotr2.shape[1]
    scale_xyz_dim = sca2.shape[1]
    f_dc_dim = fdc2.shape[1]
    f_rest_dim = frs2.shape[1]
    opacity_dim = opa2.shape[1]
    t_dim = t2.shape[1]
    scale_t_dim = st2.shape[1]

    with open(out_path, "wb") as f:
        write_header_v2(
            f=f,
            N=N,
            active_sh_degree=active_sh_degree,
            active_sh_degree_t=active_sh_degree_t,
            rot_4d=rot_4d,
            store_scale_log=store_scale_log,
            xyz_dim=xyz_dim,
            rotation_dim=rotation_dim,
            rotation_r_dim=rotation_r_dim,
            scale_xyz_dim=scale_xyz_dim,
            f_dc_dim=f_dc_dim,
            f_rest_dim=f_rest_dim,
            opacity_dim=opacity_dim,
            t_dim=t_dim,
            scale_t_dim=scale_t_dim,
        )

        for i in range(N):
            row = np.concatenate([
                xyz2[i],
                rot2[i],
                rotr2[i],
                sca2[i],
                fdc2[i],
                frs2[i],
                opa2[i],
                t2[i],
                st2[i],
            ]).astype(np.float32)
            f.write(row.tobytes(order="C"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="chkpnt_best.pth")
    ap.add_argument("--out", required=True, help="output .splat4d")
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
    args = ap.parse_args()

    x = torch.load(args.ckpt, map_location="cpu")
    if not (isinstance(x, (tuple, list)) and len(x) == 2):
        raise TypeError("Expected checkpoint format: (model_params, iteration)")
    model_params, it = x

    if not (isinstance(model_params, (tuple, list)) and len(model_params) == 19):
        raise TypeError("Expected model_params capture() format len == 19")

    (
        active_sh_degree,
        xyz, f_dc, f_rest,
        scaling_xyz, rotation, opacity,
        max_radii2D, xyz_grad_accum, t_grad_accum, denom,
        opt_state, spatial_lr_scale,
        t, scaling_t, rotation_r, rot_4d, env_map, active_sh_degree_t
    ) = model_params

    active_sh_degree = to_scalar_int(active_sh_degree, "active_sh_degree")
    active_sh_degree_t = to_scalar_int(active_sh_degree_t, "active_sh_degree_t")
    rot_4d = to_scalar_bool(rot_4d, "rot_4d")

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
