# 4dgs-viewer

This repository is a fork of GaussianSplats3D for building an interactive **true 4DGS viewer** in the browser.

## Status

This repository is **under active development**.

The goal of this fork is a **true 4DGS viewer** for interactive browser-based inspection, not a frame-by-frame sequence of independent 3DGS scenes.
However, the exporter, `.splat4d` format, and browser demo viewer are still evolving, and the current implementation should be considered **experimental**.
Breaking changes may occur as the format and rendering pipeline are refined.

## This fork

The goal of this fork is **not** a frame-by-frame sequence of independent 3DGS scenes.  
The goal is a **true 4DGS viewer** that can:

- inspect 4DGS data interactively with the mouse
- render time-aware splats in the browser
- preserve data needed for true 4DGS playback, including:
  - `xyz`
  - `rotation`
  - `rotation_r`
  - `scaling_xyz`
  - `f_dc`
  - `f_rest`
  - `opacity`
  - `t`
  - `scaling_t`

## Repository structure

- `demo/` : browser viewers and local sample files
- `converter/` : checkpoint to `.splat4d` converter
- `demo/scene_v2.splat4d` : local sample v2 scene file (may be excluded from Git because GitHub rejects files larger than 100 MB)

## Checkpoint conversion

`chkpnt_best.pth` can first be produced by training with a 4DGS implementation such as `fudan-zvg/4d-gaussian-splatting`, and then converted here using the exporter below.

Use the converter from a 4DGS environment that can load the checkpoint.

### v2 format (recommended)

```bash
python converter/export_splat4d_from_ckpt.py --ckpt chkpnt_best.pth --out demo/scene_v2.splat4d --store_scale_log
```

This exports raw data for the true 4DGS viewer, including:

- `xyz`
- `rotation`
- `rotation_r`
- `scaling_xyz`
- `f_dc`
- `f_rest`
- `opacity`
- `t`
- `scaling_t`

### legacy v1 format

```bash
python converter/export_splat4d_from_ckpt.py --ckpt chkpnt_best.pth --out demo/scene_legacy.splat4d --store_scale_log --legacy_v1
```

This is only for older compact viewers.

## Demo viewer

The current recommended local viewer is the final CPU-based viewer in `demo/`:

- `4dgs_native_cpu_viewer.html`

Place a `.splat4d` file in `demo/`, run the local demo server from the repository root, and open the page below.

```bash
npm run build
npm run demo
```

Then open:

```text
http://127.0.0.1:8080/demo/4dgs_native_cpu_viewer.html
```

### Notes on the current CPU viewer

- It is still experimental.
- It currently uses the covariance construction fixed in this repository:
  - `M = R * S`
  - `Sigma = M * M^T`
- `prefilter var` increases time-direction blur. For sharp inspection, `prefilter var = 0` is recommended.
- `stride=1` gives the closest result to the current CPU implementation target, but it can be very slow for large scenes.
- The current sample checkpoint had `active_sh_degree = 0`, so SH support is implemented in the pipeline, but the visible result for that sample is still largely DC-color based.
- This is a CPU viewer, so playback can become frame-by-frame for multi-million-gaussian scenes.

## Large files

GitHub rejects files larger than 100 MB.  
If `scene_v2.splat4d` exceeds that limit, keep it locally and do not track it with Git.

## Notes

- The current browser viewer is experimental.
- The v2 converter format should be used as the main format going forward.
- The current recommended browser demo is the CPU-based `4dgs_native_cpu_viewer.html`.

## Original 3DGS Viewer

See the original GaussianSplats3D project:

- https://github.com/mkkellogg/GaussianSplats3D
