import { createAdoptedBundleUrlReader, loadVerifiedAdoptedBundle } from './common_4dgs_adopted_bundle.js';
import { activateNormalAdoptedScene, normalStartupInput } from './common_4dgs_normal_on.js';

export function createViewerFileIO({
  ui,
  parseArrayBuffer,
  onSceneLoaded,
  scheduleRender,
  defaultSceneUrl = './scene_v2.splat4d'
}) {
  let loadGeneration = 0;
  async function publish(raw, sourceMetadata, generation) {
    if (generation !== loadGeneration) throw new Error('scene-load-superseded');
    await onSceneLoaded(raw, sourceMetadata);
    return raw;
  }

  async function loadAdoptedBundle({ baseUrl, authority, selection, readResource, normalProfile = null } = {}) {
    const generation = ++loadGeneration;
    const raw = await loadVerifiedAdoptedBundle({ authority, selection,
      readResource: readResource ?? createAdoptedBundleUrlReader(baseUrl), parseArrayBuffer });
    raw.assetPath = baseUrl ?? 'adopted-bundle';
    raw.sourcePath = raw.assetPath;
    if (normalProfile !== null) activateNormalAdoptedScene(raw, normalProfile, {baseUrl,authority,selection});
    return publish(raw, { sourceKind: raw.assetSourceKind }, generation);
  }
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

  async function decodeAndPublish(buf, sourceMetadata, generation) {
    const raw = parseArrayBuffer(buf);
    const sha256 = await sha256ArrayBuffer(buf);
    raw.assetPath = sourceMetadata.path ?? sourceMetadata.sourcePath ?? raw.assetPath ?? null;
    raw.sourcePath = sourceMetadata.sourcePath ?? sourceMetadata.path ?? raw.sourcePath ?? null;
    raw.assetSourceKind = sourceMetadata.sourceKind ?? raw.assetSourceKind ?? null;
    raw.assetSizeBytes = Number.isFinite(sourceMetadata.sizeBytes)
      ? Number(sourceMetadata.sizeBytes)
      : (Number.isFinite(buf?.byteLength) ? Number(buf.byteLength) : null);
    raw.assetSha256 = sha256 ?? raw.assetSha256 ?? null;
    return publish(raw, sourceMetadata, generation);
  }

  async function loadArrayBuffer(buf, sourceMetadata = {}) {
    return decodeAndPublish(buf, sourceMetadata, ++loadGeneration);
  }

  async function loadDefaultScene() {
    const generation = ++loadGeneration;
    try {
      const res = await fetch(defaultSceneUrl);
      if (!res.ok) return null;
      return await decodeAndPublish(await res.arrayBuffer(), {
        path: defaultSceneUrl,
        sourcePath: defaultSceneUrl,
        sourceKind: 'default-scene-url'
      }, generation);
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
      const generation = ++loadGeneration;
      await decodeAndPublish(await f.arrayBuffer(), {
        path: f.name,
        sourcePath: f.name,
        sourceKind: 'file-input',
        sizeBytes: f.size
      }, generation);
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
      const generation = ++loadGeneration;
      await decodeAndPublish(await f.arrayBuffer(), {
        path: f.name,
        sourcePath: f.name,
        sourceKind: 'drag-and-drop',
        sizeBytes: f.size
      }, generation);
      if (scheduleRender) {
        scheduleRender();
      }
    });
  }

  return {
    loadStartup: profile => profile == null ? loadDefaultScene() : loadAdoptedBundle(normalStartupInput(profile)),
    getLoadGeneration: () => loadGeneration,
    loadAdoptedBundle,
    loadArrayBuffer,
    loadDefaultScene,
    bindFileInput,
    bindDragAndDrop
  };
}
