# Phase 3 Step17 Prefix-Sum / Scatter Validation Unit

## Goal

Step17 keeps WebGPU tile-list generation deferred. The purpose is to split
prefix-sum and scatter into validation units before implementing WGSL passes.
This preserves the Step15/16 tile-list contracts while making Step18 small
enough to debug from saved dry-run JSON.

The normal backend target remains WebGPU-only. WebGL2 stays as fallback,
validation, and regression oracle. This step does not introduce hybrid drawing.

## Candidate Comparison

- A. Docs-first validation unit contract: useful because it captures the
  dependency graph and failure boundaries without changing runtime behavior.
- B. tileCounts -> tileOffsets dry-run metadata: useful as the Step18 starting
  point, but Step17 should first name the validation units that make it safe.
- C. Scatter output schema metadata: already covered by Step15 and Step16 at
  the schema level; the next useful step is separating scatter from offsets.
- D. WebGPU prefix-sum experiment: too early for Step17 because capacity and
  validation need a bounded test unit first.
- E. Sort / depth ordering contract: important later, but tile-list generation
  must first produce trustworthy per-tile lists.

Adopted approach: A with minimal B-oriented metadata. The dry-run JSON now
contains `tileListValidationUnitContract`, and Step18 should implement only
the `tileCounts-to-tileOffsets-dry-run` unit.

## Dependency Order

1. Visible records provide `tileRange`.
2. Tile grid metadata provides `tileCols`, `tileRows`, and `tileCount`.
3. The counts pass builds `tileCounts[tileCount]`.
4. The exclusive prefix pass builds `tileOffsets[tileCount + 1]`.
5. Capacity classification derives `totalTileRefs`, `maxTileRefs`, and
   `capacityStatus`.
6. Scatter uses `tileRange`, `tileOffsets`, and per-tile write cursors to fill
   `tileIndices`.
7. Tile-list metadata summarizes `totalTileRefs`, `maxRefsPerTile`,
   `nonEmptyTiles`, and validation status.

## Validation Units

- `tileCounts-from-tileRange`: validates tileRange iteration, tile grid clamp,
  and per-tile counts.
- `tileOffsets-from-tileCounts`: validates exclusive prefix sum, monotonic
  offsets, and `tileOffsets[tileCount] == sum(tileCounts)`.
- `scatter-indices-from-tileRange-and-offsets`: validates write cursor
  initialization, per-tile ordering, and `tileIndices` capacity bounds.
- `tileListMetadata-from-counts-offsets-scatter`: validates summary values and
  capacity/overflow classification.

## Failure Boundaries

- `tileCountsMismatch`: inspect tileRange generation, inclusive min/max policy,
  tile grid clamp, and visible record ordering.
- `prefixOffsetsMismatch`: inspect exclusive prefix implementation, tileCount
  bounds, and integer overflow assumptions.
- `capacityMismatch`: inspect `totalTileRefs`, `maxTileRefs`, overflow policy,
  and resize/second-pass decision.
- `scatterMismatch`: inspect write cursor initialization, atomic/write ordering,
  per-tile visible order, and tileIndices bounds.

## Contracts To Keep Fixed First

- `tileRangeContract`: inclusive tile min/max range and tile grid clamp.
- `tileListContract`: `tileCounts`, `tileOffsets`, `tileIndices`, and metadata
  output schema.
- `tileListCapacityContract`: capacity status and overflow policy.
- `tileListValidationContract`: summary schema and validation field names.
- `comparisonContract`: record comparison shape remains unchanged until
  tile-list outputs become compared buffers.

## Step18 Minimal Implementation

Step18 should implement a dry-run unit for `tileCounts -> tileOffsets` only.
It should compare WebGPU or staged output against CPU `buildTileLists` counts
and offsets, keep scatter deferred, and report:

- `tileCountsValid`
- `prefixOffsetsValid`
- `totalTileRefsConsistent`
- `capacityStatus`
- `firstValidationFailures`

Success for Step18 should not require display connection, sort, compaction,
tile composite changes, or scatter output validation.
