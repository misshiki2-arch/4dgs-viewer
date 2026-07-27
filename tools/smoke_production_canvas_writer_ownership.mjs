import assert from 'node:assert/strict';

import {
  shouldAllowDiagnosticCanvasPresentation
} from '../demo/js/webgpu_backend_frame_prototype.js';
import {
  WEBGPU_TILE_COMPOSITOR_FRAME_IMPLEMENTATION_MODE
} from '../demo/js/webgpu_tile_compositor_frame_implementation.js';
import {
  buildWebGpuViewerCanvasBoundedFirstPresent
} from '../demo/js/webgpu_viewer_canvas_bounded_first_present.js';
import {
  buildWebGpuViewerCanvasBoundedColorPresent
} from '../demo/js/webgpu_viewer_canvas_bounded_color_present.js';

const canvas = {
  width: 1280,
  height: 720,
  getContextCalls: 0,
  getContext() {
    this.getContextCalls += 1;
    return null;
  }
};
const viewerCanvasState = {
  provided: true,
  canvas,
  requestedBackendMode: 'webgpu-exclusive',
  allowViewerCanvasPresentation: true,
  webgl2FrameLifecycleSuppressed: true
};
const currentTexturePath = {
  viewerCanvasCurrentTexturePathReady: true,
  textureFormat: 'rgba8unorm'
};

assert.equal(
  shouldAllowDiagnosticCanvasPresentation(
    WEBGPU_TILE_COMPOSITOR_FRAME_IMPLEMENTATION_MODE
  ),
  false
);
assert.equal(
  shouldAllowDiagnosticCanvasPresentation('prototype-diagnostic-backend'),
  true
);

const firstPresent = await buildWebGpuViewerCanvasBoundedFirstPresent({
  device: {},
  viewerCanvasState,
  webgpuViewerCanvasCurrentTexturePath: currentTexturePath,
  diagnosticCanvasPresentationAllowed: false
});
assert.equal(firstPresent.commandBufferSubmitted, false);
assert.equal(canvas.getContextCalls, 0);

const colorPresent = await buildWebGpuViewerCanvasBoundedColorPresent({
  device: {},
  viewerCanvasState,
  webgpuViewerCanvasCurrentTexturePath: currentTexturePath,
  webgpuViewerCanvasBoundedFirstPresent: {
    boundedViewerCanvasFirstPresentSucceeded: true
  },
  webgpuViewerCanvasBoundedColorSourceSelector: {
    selectedColorSamples: [
      { samplePx: { x: 64, y: 96 }, colorAlpha: [0.25, 0.5, 0.75, 1] }
    ],
    selectedColorSource: 'ownership-smoke'
  },
  diagnosticCanvasPresentationAllowed: false
});
assert.equal(colorPresent.commandBufferSubmitted, false);
assert.equal(colorPresent.colorPresentSampleCount, 1);
assert.equal(colorPresent.vertexCount, 6);
assert.equal(canvas.getContextCalls, 0);

globalThis.GPUTextureUsage = {
  RENDER_ATTACHMENT: 1,
  COPY_SRC: 2,
  COPY_DST: 4
};
const diagnosticCanvas = {
  width: 64,
  height: 64,
  getContextCalls: 0,
  getContext() {
    this.getContextCalls += 1;
    return {
      configure() {},
      getCurrentTexture() {
        return { createView() { return {}; } };
      }
    };
  }
};
const diagnosticDevice = {
  queue: { submit() {} },
  createCommandEncoder() {
    return {
      beginRenderPass() { return { end() {} }; },
      finish() { return {}; }
    };
  }
};
const diagnosticFirstPresent = await buildWebGpuViewerCanvasBoundedFirstPresent({
  device: diagnosticDevice,
  viewerCanvasState: { ...viewerCanvasState, canvas: diagnosticCanvas },
  webgpuViewerCanvasCurrentTexturePath: currentTexturePath,
  diagnosticCanvasPresentationAllowed: shouldAllowDiagnosticCanvasPresentation(
    'prototype-diagnostic-backend'
  )
});
assert.equal(diagnosticFirstPresent.commandBufferSubmitted, true);
assert.equal(diagnosticCanvas.getContextCalls, 1);

console.log('production canvas writer ownership smoke test passed');
