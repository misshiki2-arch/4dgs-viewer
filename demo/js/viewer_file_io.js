export function createViewerFileIO({
  ui,
  parseArrayBuffer,
  onSceneLoaded,
  scheduleRender,
  defaultSceneUrl = './scene_v2.splat4d'
}) {
  async function sha256ArrayBuffer(buf) {
    if (
      typeof crypto === 'undefined' ||
      !crypto.subtle ||
      !buf ||
      typeof buf.byteLength !== 'number'
    ) {
      return null;
    }
    const digest = await crypto.subtle.digest('SHA-256', buf);
    return Array.from(new Uint8Array(digest))
      .map((byte) => byte.toString(16).padStart(2, '0'))
      .join('');
  }

  async function loadArrayBuffer(buf, sourceMetadata = {}) {
    const raw = parseArrayBuffer(buf);
    const sha256 = await sha256ArrayBuffer(buf);
    raw.assetPath = sourceMetadata.path ?? sourceMetadata.sourcePath ?? raw.assetPath ?? null;
    raw.sourcePath = sourceMetadata.sourcePath ?? sourceMetadata.path ?? raw.sourcePath ?? null;
    raw.assetSourceKind = sourceMetadata.sourceKind ?? raw.assetSourceKind ?? null;
    raw.assetSizeBytes = Number.isFinite(sourceMetadata.sizeBytes)
      ? Number(sourceMetadata.sizeBytes)
      : (Number.isFinite(buf?.byteLength) ? Number(buf.byteLength) : null);
    raw.assetSha256 = sha256 ?? raw.assetSha256 ?? null;
    await onSceneLoaded(raw, sourceMetadata);
    return raw;
  }

  async function loadDefaultScene() {
    try {
      const res = await fetch(defaultSceneUrl);
      if (!res.ok) return null;
      return await loadArrayBuffer(await res.arrayBuffer(), {
        path: defaultSceneUrl,
        sourcePath: defaultSceneUrl,
        sourceKind: 'default-scene-url'
      });
    } catch (e) {
      console.warn(e);
      return null;
    }
  }

  function bindFileInput() {
    if (!ui.fileInput) return;
    ui.fileInput.addEventListener('change', async (e) => {
      const f = e.target.files[0];
      if (!f) return;
      await loadArrayBuffer(await f.arrayBuffer(), {
        path: f.name,
        sourcePath: f.name,
        sourceKind: 'file-input',
        sizeBytes: f.size
      });
    });
  }

  function bindDragAndDrop(doc = document) {
    doc.addEventListener('dragover', (e) => {
      e.preventDefault();
      if (ui.drop) ui.drop.style.display = 'flex';
    });

    doc.addEventListener('dragleave', (e) => {
      e.preventDefault();
      if (ui.drop) ui.drop.style.display = 'none';
    });

    doc.addEventListener('drop', async (e) => {
      e.preventDefault();
      if (ui.drop) ui.drop.style.display = 'none';
      const f = e.dataTransfer.files[0];
      if (!f) return;
      await loadArrayBuffer(await f.arrayBuffer(), {
        path: f.name,
        sourcePath: f.name,
        sourceKind: 'drag-and-drop',
        sizeBytes: f.size
      });
      if (scheduleRender) {
        scheduleRender();
      }
    });
  }

  return {
    loadArrayBuffer,
    loadDefaultScene,
    bindFileInput,
    bindDragAndDrop
  };
}
