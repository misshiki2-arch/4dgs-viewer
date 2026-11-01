# AGENTS.md

## Project purpose
This repository is an experimental browser-based true 4DGS viewer built on Three.js.
The current main viewer entry is `4dgs_gpu_viewer.html`.
The long-term development goal is to evolve the GPU path step by step without breaking the current viewer.

## Program-wide purpose boundary

- The program-wide research goal and project roles are owned by [`50_4DGS_RESEARCH_PROGRAM_GOALS_JA.md`](../4dgs-development-governance/50_4DGS_RESEARCH_PROGRAM_GOALS_JA.md); the project purpose above is Viewer-specific.
- The Viewer consumes and validates direct-conversion results. It is neither the program's central research result nor the owner of the direct converter.
- Keep the direct converter separate and connect it through explicit canonical format, manifest, provenance, particle identity, time, and physical-attribute interfaces.
- Viewer Phase and Step identifiers remain local to this repository and must not be reused for the direct converter or the research program.
- LOD, streaming, and compression are not current mandatory implementations. Re-evaluate them as separately authorized responsibilities only if measured real-time use does not hold.

## Scope of this file
This file defines repository-wide rules.
Use this file for project-level instructions that should remain valid across multiple steps.
Use more local `AGENTS.md` files for directory-specific implementation rules.

## Current freeze and future-design boundary
- The canonical current-state record is `docs/phase3_current_state.md`.
- The user accepted Corrected Step21 function and Gate D input acceptance for the adopted bundle on 2026-10-25 JST. [CURRENT_STATE §24](docs/phase3_current_state.md#corrected-b-handoff) owns the implemented intake/SH0 connection, CPU/browser evidence and bounded acceptance. Adopted B and Gates A–C remain accepted.
- The Viewer remains generally frozen partway through Step122. Step21 acceptance alone neither renumbers Viewer work nor resumes Step122/Phase4. Document review, user research Git/push and Step21 completion processing are complete. Corrected is complete within its adopted scope, including the Phase5 handoff; remaining Viewer duties belong to this project and do not count as unfinished Corrected work. Preserve legacy SH2 and the shared 128-byte staging contract.
- Step122 adopted-B fixed-range numerical/image alignment was user-accepted on 2026-10-30 JST; the implemented `adopted-b-fixed-range` normal-ON startup/capture/comparison was functionally user-accepted on 2026-11-01 JST for the same fixed 65,536 records. [CURRENT_STATE §24](docs/phase3_current_state.md#step122-adopted-range) owns scope, evidence and pending decisions; the [backend contract](docs/phase3_webgpu_backend_design.md#step122-adopted-compositing) owns numerical/state/policy/frame semantics. Step122-wide completion remains subject to advisor document review and existing completion conditions. This Documentation Sync authorizes no further source changes or source switching, execution, or Git write.
- All work outside that accepted fixed scope remains frozen, including unrelated sort/compositor changes, full-scene/performance, scalability, LOD, streaming and higher/temporal SH activation. Do not infer a new Phase/Step or general unfreeze from this scope.
- Preserve the already-completed infrastructure from Phase 3 and Steps 118–122 work, including production/diagnostic device separation and production readback-none.
- `docs/phase4_and_later_scalability_design.md` records future Viewer design candidates; it does not start Phase 4, lift the freeze, or authorize implementation.
- External methods named in that document are unadopted investigation or design-spike candidates, not implementation contracts.
- Do not carry a Phase number, ticket, or design from another project into this Viewer project.
- Documentation Sync is limited to the expressly named documents and remains separate from source implementation.

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
