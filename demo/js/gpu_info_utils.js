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
  extraLines = []
}) {
  const lines = [];

  lines.push(`format=v2`);
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
    lines.push(`active_sh_degree=0  active_sh_degree_t=0`);
    lines.push(`rot_4d(file)=false  useRot4d=${useRot4d}  useSH=${useSH}`);
  }

  lines.push(
    `nativeRot4d=${useNativeRot4d}  nativeMarginal=${useNativeMarginal}`
  );
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
    lines.push(
      `tileCols=${tileSummary.tileCols}  tileRows=${tileSummary.tileRows}  nonEmptyTiles=${tileSummary.nonEmptyTiles}`
    );
    lines.push(
      `totalTileRefs=${tileSummary.totalRefs.toLocaleString()}  avgRefsPerVisible=${avgRefsPerVisible !== null ? Number(avgRefsPerVisible).toFixed(2) : '0.00'}`
    );
    lines.push(
      `avgPerNonEmptyTile=${Number(tileSummary.avgPerNonEmptyTile).toFixed(2)}  maxPerTile=${tileSummary.maxPerTile}`
    );
    lines.push(`activeTileBox=${tileSummary.activeTileBoxText}`);
    lines.push(
      `offsetsLen=${tileSummary.offsetsLen.toLocaleString()}  indicesLen=${tileSummary.indicesLen.toLocaleString()}`
    );
    lines.push(
      `countEnergy=${tileSummary.countEnergy.toLocaleString()}  ${tileSummary.sampleTileText}`
    );
    lines.push('');
  }

  if (extraLines && extraLines.length > 0) {
    for (const line of extraLines) {
      lines.push(line);
    }
  }

  return lines.join('\n');
}

export function setInfoText(infoEl, text) {
  if (!infoEl) return;
  infoEl.textContent = text;
}
