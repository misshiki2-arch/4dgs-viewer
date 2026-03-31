export function createViewerFileIO({
  ui,
  parseArrayBuffer,
  onSceneLoaded,
  scheduleRender,
  defaultSceneUrl = './scene_v2.splat4d'
}) {
  async function loadArrayBuffer(buf) {
    const raw = parseArrayBuffer(buf);
    await onSceneLoaded(raw);
    return raw;
  }

  async function loadDefaultScene() {
    try {
      const res = await fetch(defaultSceneUrl);
      if (!res.ok) return null;
      return await loadArrayBuffer(await res.arrayBuffer());
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
      await loadArrayBuffer(await f.arrayBuffer());
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
      await loadArrayBuffer(await f.arrayBuffer());
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
