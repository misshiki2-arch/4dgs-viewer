# AGENTS.md

## Project purpose
This repository is an experimental browser-based true 4DGS viewer built on Three.js.
The current main viewer entry is `4dgs_gpu_viewer.html`.
The development goal is to evolve the GPU path step by step without breaking the current viewer.

## Scope of this file
This file defines repository-wide rules.
Use this file for project-level instructions that should remain valid across multiple steps.
Use more local `AGENTS.md` files for directory-specific implementation rules.

## Main architecture principles
- Keep `viewer_app_gpu.js` thin. It is a wiring and orchestration file, not the place for new rendering logic.
- Keep rendering math, screen-space build, draw path policy, draw execution, and debug formatting separated.
- Treat fallback as an independent path, not as an unnamed `else` branch.
- Keep candidate generation separated by method.
- Do not mix debug-only logic into the main render path unless strictly necessary.

## Editing policy
- Prefer incremental changes over broad refactors.
- Default to one-file-at-a-time changes unless explicitly requested otherwise.
- Preserve public APIs unless explicitly requested to change them.
- Preserve current working behavior first, then improve internal boundaries.
- Do not rename files, exported functions, or public return fields unless explicitly requested.
- Do not rewrite unrelated files.

## Validation policy
When making a code change:
- keep the viewer runnable
- preserve existing contracts unless explicitly requested otherwise
- preserve existing return shapes unless explicitly requested otherwise
- avoid behavior changes and structure changes in the same step when possible
- report possible side effects after edits

## Git policy
- Do not stage, commit, or push unless explicitly requested.
- Treat `.codex/` as local-only state unless explicitly requested otherwise.
- Ignore unrelated untracked local files unless they are directly relevant to the task.

## Communication policy
- For design work, start with explanation or plan mode before editing.
- For implementation work, keep edits as small as possible.
- If a task would require touching many files, first propose the file list and edit order.
- After editing, report changed files, summary of changes, and possible side effects.
