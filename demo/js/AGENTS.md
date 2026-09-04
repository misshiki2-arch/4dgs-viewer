## Current Project State

- The canonical current-state record is ../../docs/phase3_current_state.md.
- The Viewer is frozen partway through Step 122 until the corrected training baseline and restart gates are accepted. There is no currently authorized Viewer implementation Step.
- During the freeze, do not change source under `demo/js`.
- Preserve the already-completed infrastructure from Phase 3 and Steps 118–122 work, including production/diagnostic device separation and production readback-none.
- The Phase 4 scalability document is future design only; it does not start Phase 4, lift the freeze, or authorize source work here.
- Any older current goal or next step statement in this file is historical guidance, not authorization to resume that numbered Step.
- Do not choose or implement a new Step from stale text. The user and design-management conversation must first fix its objective, success condition, non-goals, owner, preserved contracts, and stop conditions.
- Unless explicitly asked to update design documentation, CODEX should report design impact and leave current-state and next-Step document changes to the design-management workflow.

# AGENTS.md

## Scope of this file
This file defines local rules for `demo/js`.
These rules are more specific than the repository root `AGENTS.md` and apply to viewer implementation work in this directory.

## Current main viewer context
- The current main viewer entry is `4dgs_gpu_viewer.html`.
- This directory contains the current GPU viewer implementation and historical Step35 responsibility guidance.

## Historical Step35 implementation guidance
- These rules record Step35's boundary discipline; they do not authorize resuming Step35.
- Step35 was implemented incrementally.
- Prefer boundary fixing before deeper optimization.
- Avoid broad refactors.
- Keep one-file-at-a-time edits by default.
- Preserve current behavior whenever possible.
- Do not infer a current implementation stage from this historical guidance.

## Preferred responsibility boundaries
- `viewer_app_gpu.js`: app wiring only
- `gpu_renderer.js`: frame orchestration only
- `gpu_draw_path_selector.js`: draw path policy only
- draw executors: execution only
- `gpu_visible_builder.js`: visible evaluation only
- `gpu_screen_space_builder.js`: screen-space and packed build
- debug/info modules: formatting and display only

## Historical Step35 responsibility direction
- Keep `viewer_app_gpu.js` thin.
- Treat fallback as an independent path.
- Keep candidate generation separated by method.
- Separate candidate policy from visible evaluation.
- Separate visible evaluation from source-item and screen-space build.
- Keep debug assembly out of the main render path where possible.

## External Phase 4 candidate boundary
- External renderer, codec, and streaming methods in the Phase 4 design are unadopted investigation candidates.
- Do not trial or integrate them into the current production path during the freeze.
- Implementation may proceed only after the freeze is explicitly lifted and a new Step is explicitly authorized.

## Editing rules for this directory
- Preserve `getVisibleBuildConfig(...)` contract unless explicitly requested otherwise.
- Preserve `buildVisibleSplats(...)` contract unless explicitly requested otherwise.
- Preserve existing return shapes unless explicitly requested otherwise.
- Do not change draw/debug output fields unless explicitly requested.
- Do not mix draw path policy and draw execution in the same module unless explicitly requested.
- Do not move multiple boundaries at once unless explicitly requested.

## Historical Step35 order guidance
After the freeze is explicitly lifted and a new Step explicitly reuses this guidance:
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
