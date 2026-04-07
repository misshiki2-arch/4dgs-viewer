# AGENTS.md

## Scope of this file
This file defines local rules for `demo/js`.
These rules are more specific than the repository root `AGENTS.md` and apply to viewer implementation work in this directory.

## Current main viewer context
- The current main viewer entry is `4dgs_gpu_viewer.html`.
- This directory contains the current GPU viewer implementation and related Step35 work.
- The current development goal is Step35: fix boundaries first, then evolve the GPU path incrementally.

## Step35 implementation policy
- Implement Step35 incrementally.
- Prefer boundary fixing before deeper optimization.
- Avoid broad refactors.
- Keep one-file-at-a-time edits by default.
- Preserve current behavior whenever possible.
- Do not jump ahead to later Step35 stages unless explicitly requested.

## Preferred responsibility boundaries
- `viewer_app_gpu.js`: app wiring only
- `gpu_renderer.js`: frame orchestration only
- `gpu_draw_path_selector.js`: draw path policy only
- draw executors: execution only
- `gpu_visible_builder.js`: visible evaluation only
- `gpu_screen_space_builder.js`: screen-space and packed build
- debug/info modules: formatting and display only

## Current Step35 direction
- Keep `viewer_app_gpu.js` thin.
- Treat fallback as an independent path.
- Keep candidate generation separated by method.
- Separate candidate policy from visible evaluation.
- Separate visible evaluation from source-item and screen-space build.
- Keep debug assembly out of the main render path where possible.

## Editing rules for this directory
- Preserve `getVisibleBuildConfig(...)` contract unless explicitly requested otherwise.
- Preserve `buildVisibleSplats(...)` contract unless explicitly requested otherwise.
- Preserve existing return shapes unless explicitly requested otherwise.
- Do not change draw/debug output fields unless explicitly requested.
- Do not mix draw path policy and draw execution in the same module unless explicitly requested.
- Do not move multiple boundaries at once unless explicitly requested.

## Step order guidance
When asked to implement a Step35 change:
1. explain the smallest safe boundary to edit
2. identify the single file to touch first
3. keep the first edit minimal
4. validate behavior before moving to the next file

## Reporting rules
After each edit in this directory, report:
1. changed files
2. summary of changes
3. possible side effects
4. whether public contracts were preserved
