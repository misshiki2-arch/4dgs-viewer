const VIEWER_CAMERA_PRESETS = Object.freeze([
  {
    name: 'fit',
    label: 'Fit',
    description: 'scene-fit reset camera'
  },
  {
    name: 'diag',
    label: 'Diag',
    position: [40, 20, 20],
    target: [35, 20, 2],
    description: 'default diagonal overview'
  },
  {
    name: 'front',
    label: 'Front',
    position: [35, 20, 48],
    target: [35, 20, 2],
    description: 'front-facing comparison view'
  },
  {
    name: 'side',
    label: 'Side',
    position: [76, 20, 2],
    target: [35, 20, 2],
    description: 'side profile comparison view'
  },
  {
    name: 'top',
    label: 'Top',
    position: [35, 68, 2],
    target: [35, 20, 2],
    up: [0, 0, -1],
    description: 'top-down comparison view'
  }
]);

export function getViewerCameraPresets() {
  return VIEWER_CAMERA_PRESETS.slice();
}

export function resolveViewerCameraPreset(name) {
  if (typeof name !== 'string' || name.length === 0) return null;
  const normalized = name.trim().toLowerCase();
  return VIEWER_CAMERA_PRESETS.find((preset) => preset.name === normalized) ?? null;
}
