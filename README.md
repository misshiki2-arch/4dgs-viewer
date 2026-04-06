# 4dgs-viewer

This repository is a fork of GaussianSplats3D for building an interactive browser viewer for **true 4D Gaussian Splatting (4DGS)**.

## Purpose

The goal of this fork is **not** a frame-by-frame sequence of independent 3DGS scenes.  
The goal is a **true 4DGS viewer** that can:

- inspect 4DGS data interactively in the browser
- render time-aware splats directly from 4DGS parameters
- preserve the parameters required for true 4D playback, including:
  - `xyz`
  - `rotation`
  - `rotation_r`
  - `scaling_xyz`
  - `f_dc`
  - `f_rest`
  - `opacity`
  - `t`
  - `scaling_t`

## Status

This repository is under active development and remains **experimental**.  
The `.splat4d` format, converter, and browser viewer are still evolving.

## Repository structure

- `demo/` : browser demo files
- `demo/js/` : viewer-side JavaScript modules
- `converter/` : checkpoint to `.splat4d` converter
- `src/` : original library-side code from the upstream viewer base
- `util/` : local demo server

## Checkpoint conversion

A `.splat4d` file can be exported from a 4DGS checkpoint such as one produced by `fudan-zvg/4d-gaussian-splatting`.

Recommended v2 export:

```bash
python converter/export_splat4d_from_ckpt.py --ckpt chkpnt_best.pth --out demo/scene_v2.splat4d --store_scale_log
```

This stores the raw data needed for the true 4DGS viewer.

Legacy v1 export is still available for older compact viewers:

```bash
python converter/export_splat4d_from_ckpt.py --ckpt chkpnt_best.pth --out demo/scene_legacy.splat4d --store_scale_log --legacy_v1
```

## Local demo

Build the project and start the local demo server:

```bash
npm run build
npm run demo
```

Then open the demo from:

```text
http://127.0.0.1:8080/demo/
```

## Notes

- This project is experimental and breaking changes may occur.
- Large `.splat4d` files may exceed GitHub's 100 MB limit and should be kept locally.
- The upstream base project is GaussianSplats3D:
  - https://github.com/mkkellogg/GaussianSplats3D

## License

MIT
