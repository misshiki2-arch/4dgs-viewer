# 4dgs-viewer

This repository is a fork of GaussianSplats3D for building an interactive **true 4DGS viewer** in the browser.

## Status

This repository is **under active development**.

The goal of this fork is a **true 4DGS viewer** for interactive browser-based inspection, not a frame-by-frame sequence of independent 3DGS scenes.
However, the exporter, `.splat4d` format, and local HTML demo viewers are still evolving, and the current implementation should be considered **experimental**.
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

- `demo/` : browser viewers (`*.html`)
- `converter/` : checkpoint to `.splat4d` converter
- `demo/scene_v2.splat4d` : local sample v2 scene file (may be excluded from Git because GitHub rejects files larger than 100 MB)

## Checkpoint conversion

`chkpnt_best.pth` can first be produced by training with a 4DGS implementation such as `fudan-zvg/4d-gaussian-splatting`, and then converted here using the exporter below.

Use the converter from a 4DGS environment that can load the checkpoint.

### v2 format (recommended)

```bash
python converter/export_splat4d_from_ckpt.py   --ckpt chkpnt_best.pth   --out demo/scene_v2.splat4d   --store_scale_log
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
python converter/export_splat4d_from_ckpt.py   --ckpt chkpnt_best.pth   --out demo/scene_legacy.splat4d   --store_scale_log   --legacy_v1
```

This is only for older compact viewers.

## Demo viewers

Current experimental viewers are in `demo/`.

Current recommended viewer:

- `4dgs_covariance_splat_viewer_v2_sh_rot4d_true.html`

Other available v2 viewers:

- `4dgs_covariance_splat_viewer_v2.html`
- `4dgs_covariance_splat_viewer_v2_sh.html`

Legacy experimental viewers:

- `4dgs_gauss_viewer.html`
- `4dgs_elliptical_splat_viewer.html`
- `4dgs_covariance_splat_viewer.html`

Place a `.splat4d` file in `demo/`, run the local demo server from the repository root, and then open one of the 4DGS demo viewer pages directly.

Example:

```bash
npm run build
npm run demo
```

Then open the recommended v2 viewer in your browser:

```text
http://127.0.0.1:8080/demo/4dgs_covariance_splat_viewer_v2_sh_rot4d_true.html
```

Other available v2 demo viewer pages are:

```text
http://127.0.0.1:8080/demo/4dgs_covariance_splat_viewer_v2.html
http://127.0.0.1:8080/demo/4dgs_covariance_splat_viewer_v2_sh.html
http://127.0.0.1:8080/demo/4dgs_covariance_splat_viewer_v2_sh_rot4d_true.html
```

Legacy experimental pages are:

```text
http://127.0.0.1:8080/demo/4dgs_gauss_viewer.html
http://127.0.0.1:8080/demo/4dgs_elliptical_splat_viewer.html
http://127.0.0.1:8080/demo/4dgs_covariance_splat_viewer.html
```

Notes on the current recommended viewer:

- It is still experimental.
- `use rot4d` may improve the visual result depending on the scene.
- `stride=1` can be very slow for large scenes (for example, multi-million-gaussian scenes), and playback may become frame-by-frame.
- The current sample checkpoint had `active_sh_degree = 0`, so SH support is implemented in the viewer pipeline but the visible result is still largely DC-color based for that sample.

## Large files

GitHub rejects files larger than 100 MB.  
If `scene_v2.splat4d` exceeds that limit, keep it locally and do not track it with Git.

## Notes

- The current viewers are experimental.
- The covariance-based v2 viewer with SH and rot4d support is currently the most advanced among the local HTML demos.
- A fully faithful true 4DGS viewer requires the raw exported coefficients and rotations, so the v2 converter format should be used as the base going forward.

## Original 3DGS Viewer

For the original GaussianSplats3D project and its full 3DGS viewer documentation, see:

- https://github.com/mkkellogg/GaussianSplats3D
