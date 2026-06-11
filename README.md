# 4dgs-viewer

This repository is a fork of
[GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D) for building
an interactive browser viewer for **true 4D Gaussian Splatting (4DGS)**.

## Purpose

The goal of this fork is **not** a frame-by-frame sequence of independent 3DGS
scenes.

The goal is a **true 4DGS browser viewer** that can:

- inspect 4DGS data interactively in the browser
- render time-aware splats directly from 4DGS parameters
- preserve the parameter set needed for true 4D playback rather than reducing
  the problem to a list of per-frame 3DGS scenes

## Status

This repository is under active development and remains **experimental**.

Development is organized in phases:

- **Phase 1**: initial WebGL2 viewer and visual validation.
- **Phase 2**: WebGL2 fixed-record validation, packed-like dry-run validation,
  and a clear boundary check for what WebGL2 can do naturally versus what
  should not be forced into it. This phase is complete.
- **Phase 3**: current phase. Work has moved into a WebGPU backend prototype so
  the project can progress toward a realtime 4DGS viewer with more of the core
  pipeline on the GPU.

At the moment, the WebGPU path is still a **prototype / validation backend**.
It is not yet the finished normal viewer path.

The following are still future work:

- a production WebGPU renderer
- GPU sort / compaction
- tile-list generation
- display connection from validated records into the final rendering path

In short: the project has already gone beyond an initial browser demo, but it
is still in the middle of backend and validation work rather than at a finished
viewer release.

## Repository structure

- `demo/`: browser demo files
- `demo/js/`: viewer-side JavaScript modules, validation runtimes, and
  prototype backend scaffolding
- `converter/`: checkpoint to `.splat4d` converter
- `src/`: original library-side code from the upstream viewer base
- `tools/`: local helper scripts for deterministic URLs, capture flows, and
  saved-result summaries
- `util/`: local demo server
- `docs/`: development notes and phase summaries

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

For Phase 3 viewer/WebGPU validation, `npm run build` is the lightweight path:
it syncs `demo/` into `build/demo` without rebuilding the Rollup library bundle.
Use `npm run build:full` after changing `src/` or the published library entry
points, and use `npm run build:lib` when only the library bundle needs a Rollup
rebuild. Use `npm run rebuild` when `build/` should be deleted and recreated
from scratch.

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
