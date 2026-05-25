# Phase 3 Step19 Tile Counts / Offsets Comparison Surface

## Goal

Step19 keeps WebGPU tile-list generation deferred. Step18 already materializes
CPU reference `tileCounts` and exclusive `tileOffsets`; this step defines how a
future WebGPU `tileCounts` or `tileOffsets` result should be compared against
that reference.

No WebGPU tileCounts compute, prefix sum, scatter, sort, compaction, display
connection, or tile composite change is part of this step.

## Candidate Comparison

- A. Docs-first comparison surface: useful for naming mismatch classes and
  failure boundaries without changing runtime behavior.
- B. JSON metadata for comparison surface: useful because saved dry-run JSON can
  carry the future comparison contract alongside Step18 reference data.
- C. CPU reference self-comparison: useful as the smallest Step20 implementation
  after the surface is fixed.
- D. WebGPU upload/readback with CPU compute: not useful yet because it adds
  buffer plumbing without testing tile counting or prefix logic.
- E. WebGPU tileCounts compute: too early for Step19 because the comparison
  surface should exist before any WebGPU result is trusted.

Adopted approach: A + B. Step19 adds comparison-surface metadata and this doc.
Step20 should add CPU reference self-comparison using the same surface.

## tileCounts Comparison Surface

Compare `uint32[tileCount]` exactly.

- Shape: `tileCounts.length == tileCount`.
- Summary: `tileCountsMismatchCount`, `maxAbsCountDelta`, and
  `firstMismatches`.
- Samples: deterministic tiles such as first tile, first non-empty tile,
  max-count tile, and final tile.
- Failure boundary: tileRange generation, inclusive range iteration, tile grid
  clamp, visible filtering, or candidate/visible order.

## tileOffsets Comparison Surface

Compare `uint32[tileCount + 1]` exactly as an exclusive prefix sum.

- Shape: `tileOffsets.length == tileCount + 1`.
- Invariants: `tileOffsets[0] == 0`, monotonic offsets, and
  `tileOffsets[i + 1] - tileOffsets[i] == tileCounts[i]`.
- Summary: `tileOffsetsMismatchCount`, `maxAbsOffsetDelta`, and
  `firstMismatches`.
- Failure boundary: exclusive prefix algorithm, tileCount bounds, integer
  overflow assumptions, or mismatched tileCounts input.

## Capacity / totalTileRefs Surface

Compare capacity metadata after counts and offsets are known.

- `totalTileRefs == tileOffsets[tileCount]`.
- `totalTileRefs == sum(tileCounts)`.
- `capacityStatus` should match the expected overflow classification.
- `maxRefsPerTile` and `nonEmptyTiles` should be derived from tileCounts.

## Mismatch Classification

- `none`: no mismatch.
- `shapeMismatch`: tile count, counts length, or offsets length mismatch.
- `tileCountsMismatch`: per-tile counts differ.
- `tileOffsetsMismatch`: exclusive prefix offsets differ.
- `totalTileRefsMismatch`: terminal offset or sum mismatch.
- `capacityStatusMismatch`: overflow/capacity classification differs.

## Saved JSON Shape

Future comparison summaries should avoid dumping large duplicate buffers unless
explicitly requested. The stable surface should include:

- `anyMismatch`
- `tileCountsMismatchCount`
- `tileOffsetsMismatchCount`
- `totalTileRefsMismatch`
- `capacityStatusMismatch`
- `maxAbsCountDelta`
- `maxAbsOffsetDelta`
- `firstMismatches`
- `sampleTiles`

## Step20 Minimal Implementation

Step20 should add a CPU reference self-comparison summary. The expected and
actual sides can both come from Step18 `tileCountsToOffsetsDryRun`, proving the
comparison shape before any WebGPU tileCounts buffer exists.

Success for Step20 should mean:

- `comparisonSummary.anyMismatch == false`
- `tileCountsMismatchCount == 0`
- `tileOffsetsMismatchCount == 0`
- `totalTileRefsMismatch == false`
- `capacityStatusMismatch == false`
- `firstMismatches == []`
