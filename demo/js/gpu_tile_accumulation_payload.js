const DEFAULT_FLOATS_PER_ITEM = 16;
const FLOATS_PER_TEXEL = 4;
const ORDER_PREVIEW_LIMIT = 3;

function getBatchPackedCount(batch) {
  return Number.isFinite(batch?.packedCount) ? Math.max(0, batch.packedCount | 0) : 0;
}

function getBatchFloatsPerItem(batch) {
  return Number.isFinite(batch?.floatsPerItem)
    ? Math.max(0, batch.floatsPerItem | 0)
    : DEFAULT_FLOATS_PER_ITEM;
}

function normalizeMaxTextureSize(value) {
  const n = Number(value);
  return Number.isFinite(n) && n > 0 ? (n | 0) : 0;
}

function readPackedDepth(batch, itemIndex) {
  const packed = batch?.packed;
  const floatsPerItem = getBatchFloatsPerItem(batch);
  const base = itemIndex * floatsPerItem;
  if (!(packed instanceof Float32Array) || base < 0 || (base + 4) > packed.length) return null;
  const depth = Number(packed[base + 3]);
  return Number.isFinite(depth) ? depth : null;
}

function buildBatchPreviewEntry(batch, itemIndex) {
  if (!Number.isFinite(itemIndex) || itemIndex < 0) return null;
  return {
    localOrder: itemIndex | 0,
    sourceVisibleIndex: batch?.orderedIndices instanceof Uint32Array ? (batch.orderedIndices[itemIndex] ?? -1) : -1,
    sourceSplatIndex: batch?.sourceIndices instanceof Uint32Array ? (batch.sourceIndices[itemIndex] ?? -1) : -1,
    depth: readPackedDepth(batch, itemIndex)
  };
}

function buildBatchOrderingPreview(batch) {
  const packedCount = getBatchPackedCount(batch);
  const head = [];
  const tail = [];
  const depthHead = [];
  const depthTail = [];
  let previousDepth = null;
  let mismatchCount = 0;
  let firstMismatch = null;

  for (let i = 0; i < packedCount; i++) {
    const depth = readPackedDepth(batch, i);
    if (i < ORDER_PREVIEW_LIMIT) {
      const entry = buildBatchPreviewEntry(batch, i);
      if (entry) head.push(entry);
      depthHead.push(depth);
    }
    if (i >= Math.max(0, packedCount - ORDER_PREVIEW_LIMIT)) {
      const entry = buildBatchPreviewEntry(batch, i);
      if (entry) tail.push(entry);
      depthTail.push(depth);
    }
    if (
      Number.isFinite(previousDepth) &&
      Number.isFinite(depth) &&
      depth < previousDepth
    ) {
      mismatchCount++;
      if (!firstMismatch) {
        firstMismatch = {
          localOrder: i,
          previousDepth,
          currentDepth: depth
        };
      }
    }
    if (Number.isFinite(depth)) previousDepth = depth;
  }

  return {
    batchSpan: 1,
    sequenceConsistent: mismatchCount === 0,
    orderingMismatchCount: mismatchCount,
    firstMismatch,
    orderPreviewHead: head,
    orderPreviewTail: tail,
    depthPreviewHead: depthHead,
    depthPreviewTail: depthTail,
    firstDepth: depthHead.length > 0 ? depthHead[0] : null,
    lastDepth: depthTail.length > 0 ? depthTail[depthTail.length - 1] : null
  };
}

function buildInvalidPayloadSummary({
  batchCount,
  totalItemCount,
  maxBatchItemCount,
  payloadFloatCount = 0,
  payloadTextureWidth = 0,
  payloadTextureHeight = 0,
  payloadRowsPerColumn = 0,
  payloadColumnCount = 0,
  payloadLayoutReason = 'invalid-layout',
  payloadLayoutFailureReason = 'invalid-layout',
  maxTextureSize = 0
}) {
  return {
    payloadContract: 'tile-plan-to-accumulation-texture',
    batchCount,
    totalItemCount,
    maxBatchItemCount,
    payloadFloatCount,
    payloadTextureWidth,
    payloadTextureHeight,
    payloadRowsPerColumn,
    payloadColumnCount,
    payloadLayoutReason,
    payloadLayoutValid: false,
    payloadLayoutFailureReason,
    maxTextureSize
  };
}

function computeTextureLayout(totalItemCount, floatsPerItem, maxTextureSize) {
  const safeItems = Math.max(0, totalItemCount | 0);
  const safeFloatsPerItem = Math.max(1, floatsPerItem | 0);
  const safeMaxTextureSize = normalizeMaxTextureSize(maxTextureSize);
  const texelsPerItem = Math.max(1, Math.ceil(safeFloatsPerItem / FLOATS_PER_TEXEL));

  if (safeMaxTextureSize <= 0) {
    return buildInvalidPayloadSummary({
      batchCount: 0,
      totalItemCount: safeItems,
      maxBatchItemCount: 0,
      payloadLayoutReason: 'missing-max-texture-size',
      payloadLayoutFailureReason: 'max-texture-size-unavailable',
      maxTextureSize: safeMaxTextureSize
    });
  }

  if (texelsPerItem > safeMaxTextureSize) {
    return buildInvalidPayloadSummary({
      batchCount: 0,
      totalItemCount: safeItems,
      maxBatchItemCount: 0,
      payloadLayoutReason: 'item-wider-than-max-texture-size',
      payloadLayoutFailureReason: 'texels-per-item-exceeds-max-texture-size',
      maxTextureSize: safeMaxTextureSize
    });
  }

  if (safeItems <= 0) {
    return {
      payloadContract: 'tile-plan-to-accumulation-texture',
      batchCount: 0,
      totalItemCount: 0,
      maxBatchItemCount: 0,
      payloadFloatCount: 0,
      payloadTextureWidth: texelsPerItem,
      payloadTextureHeight: 1,
      payloadRowsPerColumn: 1,
      payloadColumnCount: 1,
      payloadLayoutReason: 'empty-payload',
      payloadLayoutValid: true,
      payloadLayoutFailureReason: 'none',
      maxTextureSize: safeMaxTextureSize
    };
  }

  const maxColumns = Math.max(1, Math.floor(safeMaxTextureSize / texelsPerItem));
  if (maxColumns <= 0) {
    return buildInvalidPayloadSummary({
      batchCount: 0,
      totalItemCount: safeItems,
      maxBatchItemCount: 0,
      payloadLayoutReason: 'no-columns-fit-max-texture-size',
      payloadLayoutFailureReason: 'max-columns-zero',
      maxTextureSize: safeMaxTextureSize
    });
  }

  const rowsPerColumn = Math.max(1, Math.min(safeMaxTextureSize, Math.ceil(safeItems / maxColumns)));
  const columnCount = Math.max(1, Math.ceil(safeItems / rowsPerColumn));
  const textureWidth = columnCount * texelsPerItem;
  const textureHeight = rowsPerColumn;

  if (
    !Number.isFinite(rowsPerColumn) ||
    !Number.isFinite(columnCount) ||
    !Number.isFinite(textureWidth) ||
    !Number.isFinite(textureHeight)
  ) {
    return buildInvalidPayloadSummary({
      batchCount: 0,
      totalItemCount: safeItems,
      maxBatchItemCount: 0,
      payloadLayoutReason: 'non-finite-layout',
      payloadLayoutFailureReason: 'non-finite-layout-dimension',
      maxTextureSize: safeMaxTextureSize
    });
  }

  if (textureWidth <= 0 || textureHeight <= 0) {
    return buildInvalidPayloadSummary({
      batchCount: 0,
      totalItemCount: safeItems,
      maxBatchItemCount: 0,
      payloadLayoutReason: 'non-positive-layout',
      payloadLayoutFailureReason: 'texture-dimension-non-positive',
      maxTextureSize: safeMaxTextureSize
    });
  }

  if (textureWidth > safeMaxTextureSize || textureHeight > safeMaxTextureSize) {
    return buildInvalidPayloadSummary({
      batchCount: 0,
      totalItemCount: safeItems,
      maxBatchItemCount: 0,
      payloadTextureWidth: textureWidth,
      payloadTextureHeight: textureHeight,
      payloadRowsPerColumn: rowsPerColumn,
      payloadColumnCount: columnCount,
      payloadLayoutReason: 'layout-exceeds-max-texture-size',
      payloadLayoutFailureReason: 'texture-dimension-exceeds-max-texture-size',
      maxTextureSize: safeMaxTextureSize
    });
  }

  return {
    payloadContract: 'tile-plan-to-accumulation-texture',
    batchCount: 0,
    totalItemCount: safeItems,
    maxBatchItemCount: 0,
    payloadFloatCount: textureWidth * textureHeight * FLOATS_PER_TEXEL,
    payloadTextureWidth: textureWidth,
    payloadTextureHeight: textureHeight,
    payloadRowsPerColumn: rowsPerColumn,
    payloadColumnCount: columnCount,
    payloadLayoutReason: columnCount > 1 ? 'multi-column-max-texture-layout' : 'single-column-layout',
    payloadLayoutValid: true,
    payloadLayoutFailureReason: 'none',
    maxTextureSize: safeMaxTextureSize,
    texelsPerItem
  };
}

function writePackedItemToTextureData(textureData, layout, itemIndex, srcFloats, srcBase, floatsPerItem) {
  const rowsPerColumn = layout.payloadRowsPerColumn;
  const texelsPerItem = layout.texelsPerItem;
  const textureWidth = layout.payloadTextureWidth;
  const columnIndex = Math.floor(itemIndex / rowsPerColumn);
  const rowIndex = itemIndex - columnIndex * rowsPerColumn;
  const texelXBase = columnIndex * texelsPerItem;

  for (let texelIndex = 0; texelIndex < texelsPerItem; texelIndex++) {
    const dstBase = ((rowIndex * textureWidth) + texelXBase + texelIndex) * FLOATS_PER_TEXEL;
    const srcTexelBase = srcBase + texelIndex * FLOATS_PER_TEXEL;
    for (let c = 0; c < FLOATS_PER_TEXEL; c++) {
      const srcOffset = srcTexelBase + c;
      textureData[dstBase + c] = srcOffset < srcBase + floatsPerItem ? srcFloats[srcOffset] : 0;
    }
  }
}

export function buildTileAccumulationPayload(tileCompositePlan, options = {}) {
  const batches = Array.isArray(tileCompositePlan?.batches) ? tileCompositePlan.batches : [];
  const usableBatches = [];
  let totalItemCount = 0;
  let maxBatchItemCount = 0;
  let maxFloatsPerItem = DEFAULT_FLOATS_PER_ITEM;

  for (const batch of batches) {
    const packedCount = getBatchPackedCount(batch);
    const floatsPerItem = getBatchFloatsPerItem(batch);
    if (!(batch?.packed instanceof Float32Array) || packedCount <= 0 || floatsPerItem <= 0) continue;

    usableBatches.push(batch);
    totalItemCount += packedCount;
    if (packedCount > maxBatchItemCount) maxBatchItemCount = packedCount;
    if (floatsPerItem > maxFloatsPerItem) maxFloatsPerItem = floatsPerItem;
  }

  const layout = computeTextureLayout(totalItemCount, maxFloatsPerItem, options?.maxTextureSize);
  layout.batchCount = usableBatches.length;
  layout.maxBatchItemCount = maxBatchItemCount;

  if (!layout.payloadLayoutValid) {
    return {
      batchMetadata: [],
      packedFloats: new Float32Array(0),
      totalItemCount,
      batchCount: usableBatches.length,
      maxBatchItemCount,
      floatsPerItem: maxFloatsPerItem,
      textureWidth: 0,
      textureHeight: 0,
      rowsPerColumn: 0,
      columnCount: 0,
      maxTextureSize: normalizeMaxTextureSize(options?.maxTextureSize),
      summary: {
        ...layout,
        batchCount: usableBatches.length,
        totalItemCount,
        maxBatchItemCount
      }
    };
  }

  const textureFloatCount = layout.payloadFloatCount;
  const textureData = new Float32Array(textureFloatCount);
  const batchMetadata = new Array(usableBatches.length);
  let itemCursor = 0;

  for (let i = 0; i < usableBatches.length; i++) {
    const batch = usableBatches[i];
    const packedCount = getBatchPackedCount(batch);
    const floatsPerItem = getBatchFloatsPerItem(batch);

    for (let itemIndex = 0; itemIndex < packedCount; itemIndex++) {
      const srcBase = itemIndex * floatsPerItem;
      writePackedItemToTextureData(textureData, layout, itemCursor + itemIndex, batch.packed, srcBase, floatsPerItem);
    }

    batchMetadata[i] = {
      tileId: batch.tileId ?? -1,
      tx: batch.tx ?? 0,
      ty: batch.ty ?? 0,
      rect: Array.isArray(batch.rect) ? batch.rect.slice(0, 4) : [0, 0, 0, 0],
      packedCount,
      floatsPerItem: maxFloatsPerItem,
      startItemIndex: itemCursor,
      ...buildBatchOrderingPreview(batch)
    };
    itemCursor += packedCount;
  }

  return {
    batchMetadata,
    packedFloats: textureData,
    totalItemCount,
    batchCount: batchMetadata.length,
    maxBatchItemCount,
    floatsPerItem: maxFloatsPerItem,
    textureWidth: layout.payloadTextureWidth,
    textureHeight: layout.payloadTextureHeight,
    rowsPerColumn: layout.payloadRowsPerColumn,
    columnCount: layout.payloadColumnCount,
    maxTextureSize: layout.maxTextureSize,
    texelsPerItem: layout.texelsPerItem,
    summary: {
      payloadContract: 'tile-plan-to-accumulation-texture',
      batchCount: batchMetadata.length,
      totalItemCount,
      maxBatchItemCount,
      payloadFloatCount: textureData.length,
      payloadTextureWidth: layout.payloadTextureWidth,
      payloadTextureHeight: layout.payloadTextureHeight,
      payloadRowsPerColumn: layout.payloadRowsPerColumn,
      payloadColumnCount: layout.payloadColumnCount,
      payloadLayoutReason: layout.payloadLayoutReason,
      payloadLayoutValid: true,
      payloadLayoutFailureReason: 'none',
      maxTextureSize: layout.maxTextureSize
    }
  };
}
