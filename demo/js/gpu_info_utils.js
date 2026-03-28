export function formatGpuViewerInfo({
  raw,
  visibleCount,
  drawCount,
  stride,
  useRot4d,
  useSH,
  useNativeRot4d,
  useNativeMarginal,
  prefilterVar,
  sigmaScale,
  renderScale,
  canvasWidth,
  canvasHeight,
  timestamp,
  splatScale,
  elapsedMs,
  stepLabel = 'GPU',
  stepNotes = [],
  tileSummary = null,
  avgRefsPerVisible = null,
  drawStats = null,
  tileSelectionText = '',
  tileDebugText = '',
  extraLines = []
}) {
  const lines = [];

  lines.push('format=v2');
  lines.push(
    `N=${raw ? raw.N.toLocaleString() : 0}  visible=${(visibleCount ?? 0).toLocaleString()}  draw=${(drawCount ?? 0).toLocaleString()}  stride=${stride}`
  );

  if (raw) {
    lines.push(
      `active_sh_degree=${raw.activeShDegree}  active_sh_degree_t=${raw.activeShDegreeT}`
    );
    lines.push(
      `rot_4d(file)=${raw.rot4d}  useRot4d=${useRot4d}  useSH=${useSH}`
    );
  } else {
    lines.push('active_sh_degree=0  active_sh_degree_t=0');
    lines.push(`rot_4d(file)=false  useRot4d=${useRot4d}  useSH=${useSH}`);
  }

  lines.push(`nativeRot4d=${useNativeRot4d}  nativeMarginal=${useNativeMarginal}`);
  lines.push(
    `prefilterVar=${Number(prefilterVar).toFixed(2)}  sigmaScale=${Number(sigmaScale).toFixed(2)}`
  );
  lines.push(
    `renderScale=${Number(renderScale).toFixed(2)}  canvas=${canvasWidth}x${canvasHeight}`
  );
  lines.push(
    `time=${Number(timestamp).toFixed(2)}  splatScale=${Number(splatScale).toFixed(2)}`
  );
  lines.push(`${stepLabel} render=${Number(elapsedMs).toFixed(1)} ms`);
  lines.push('');

  if (stepNotes && stepNotes.length > 0) {
    lines.push(`${stepLabel} note:`);
    for (const note of stepNotes) {
      lines.push(`- ${note}`);
    }
    lines.push('');
  }

  if (tileSummary) {
    lines.push('tile summary:');
    lines.push(
      `- tileCols=${tileSummary.tileCols}  tileRows=${tileSummary.tileRows}  nonEmptyTiles=${tileSummary.nonEmptyTiles}`
    );
    lines.push(
      `- totalTileRefs=${tileSummary.totalRefs.toLocaleString()}  avgRefsPerVisible=${avgRefsPerVisible !== null ? Number(avgRefsPerVisible).toFixed(2) : '0.00'}`
    );
    lines.push(
      `- avgPerNonEmptyTile=${Number(tileSummary.avgPerNonEmptyTile).toFixed(2)}  maxPerTile=${tileSummary.maxPerTile}`
    );
    lines.push(`- activeTileBox=${tileSummary.activeTileBoxText}`);
    lines.push(
      `- offsetsLen=${tileSummary.offsetsLen.toLocaleString()}  indicesLen=${tileSummary.indicesLen.toLocaleString()}`
    );
    lines.push(
      `- countEnergy=${tileSummary.countEnergy.toLocaleString()}  ${tileSummary.sampleTileText}`
    );
    lines.push('');
  }

  if (drawStats) {
    lines.push('draw stats:');
    lines.push(`- drawCount=${drawStats.drawCount}`);
    lines.push(`- visibleCount=${drawStats.visibleCount}`);
    lines.push(`- drawFraction=${Number(drawStats.drawFraction).toFixed(3)}`);
    lines.push(`- drawSelectedOnly=${drawStats.drawSelectedOnly}`);
    lines.push(`- showOverlay=${drawStats.showOverlay}`);
    lines.push(`- useMaxTile=${drawStats.useMaxTile}`);
    lines.push(`- selectedTileId=${drawStats.selectedTileId}`);
    lines.push(`- tileRadius=${drawStats.tileRadius}`);
    lines.push(`- focusTileId=${drawStats.focusTileId}`);
    lines.push(
      `- focusTileIds=${drawStats.focusTileIds && drawStats.focusTileIds.length > 0 ? '[' + drawStats.focusTileIds.join(', ') + ']' : 'none'}`
    );
    lines.push('');

    if (drawStats.tileBatchSummary) {
      lines.push('per-tile batch summary:');
      lines.push(`- tileBatchCount=${drawStats.tileBatchSummary.tileBatchCount}`);
      lines.push(`- totalTileDrawCount=${drawStats.tileBatchSummary.totalTileDrawCount}`);
      lines.push(`- maxTileDrawCount=${drawStats.tileBatchSummary.maxTileDrawCount}`);
      lines.push(`- maxTileId=${drawStats.tileBatchSummary.maxTileId}`);
      lines.push(`- avgTileDrawCount=${Number(drawStats.tileBatchSummary.avgTileDrawCount).toFixed(2)}`);
      lines.push('');
    }

    if (drawStats.executionSummary) {
      lines.push('execution summary:');
      lines.push(`- uploadCount=${drawStats.executionSummary.uploadCount}`);
      lines.push(`- drawCallCount=${drawStats.executionSummary.drawCallCount}`);
      if (typeof drawStats.executionSummary.tileBatchCount !== 'undefined') {
        lines.push(`- executionTileBatchCount=${drawStats.executionSummary.tileBatchCount}`);
      }
      lines.push('');
    }
  }

  let temporalLine = null;
  if (extraLines && extraLines.length > 0) {
    temporalLine = extraLines.find(
      line => typeof line === 'string' && line.startsWith('temporalPassed=')
    ) || null;
  }

  if (temporalLine) {
    lines.push('temporal culling summary:');
    lines.push(`- ${temporalLine}`);
    lines.push('');
  }

  if (tileSelectionText) {
    lines.push('tile selection:');
    lines.push(tileSelectionText);
    lines.push('');
  }

  if (tileDebugText) {
    lines.push('tile debug:');
    lines.push(tileDebugText);
    lines.push('');
  }

  if (extraLines && extraLines.length > 0) {
    const filteredExtra = extraLines.filter(line => line !== temporalLine);
    if (filteredExtra.length > 0) {
      lines.push('extra:');
      for (const line of filteredExtra) {
        lines.push(line);
      }
    }
  }

  return lines.join('\n');
}

export function setInfoText(infoEl, text) {
  if (!infoEl) return;
  infoEl.textContent = text;
}
