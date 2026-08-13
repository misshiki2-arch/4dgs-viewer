import assert from 'node:assert/strict';

import {
  shouldAllowDiagnosticCanvasPresentation
} from '../demo/js/webgpu_backend_frame_prototype.js';
import {
  buildProductionPresentationMutationPolicy,
  canMutateProductionPresentationState,
  PRODUCTION_DIAGNOSTIC_OBSERVER_ROLE
} from '../demo/js/common_4dgs_production_runtime_contract.js';
import {
  WEBGPU_TILE_COMPOSITOR_FRAME_IMPLEMENTATION_MODE
} from '../demo/js/webgpu_tile_compositor_frame_implementation.js';
import {
  buildWebGpuViewerCanvasBoundedFirstPresent
} from '../demo/js/webgpu_viewer_canvas_bounded_first_present.js';
import {
  buildWebGpuViewerCanvasBoundedColorPresent
} from '../demo/js/webgpu_viewer_canvas_bounded_color_present.js';
import {
  buildWebGpuViewerCanvasCurrentTexturePathReadiness
} from '../demo/js/webgpu_viewer_canvas_current_texture_path.js';
import {
  canOwnProductionTileCompositorPresentation
} from '../demo/js/webgpu_tile_list_compositor.js';

globalThis.GPUTextureUsage = {
  RENDER_ATTACHMENT: 1,
  COPY_SRC: 2,
  COPY_DST: 4
};

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
const diagnosticOwnershipPolicy = buildProductionPresentationMutationPolicy({
  executionRole: PRODUCTION_DIAGNOSTIC_OBSERVER_ROLE
});

assert.equal(diagnosticOwnershipPolicy.diagnosticComputationAllowed, true);
assert.equal(diagnosticOwnershipPolicy.diagnosticReadbackAllowed, true);
for (const field of [
  'liveProductionCanvasMutationAllowed',
  'productionOutputMutationAllowed',
  'lastValidProductionCacheMutationAllowed',
  'productionRequestMutationAllowed',
  'productionGenerationMutationAllowed',
  'compositorGenerationMutationAllowed',
  'presentedGenerationMutationAllowed',
  'finalProductionWriterMutationAllowed'
]) {
  assert.equal(diagnosticOwnershipPolicy[field], false, field);
}
assert.equal(canMutateProductionPresentationState(diagnosticOwnershipPolicy), false);
assert.equal(
  canOwnProductionTileCompositorPresentation({
    ...viewerCanvasState,
    productionPresentationMutationPolicy: diagnosticOwnershipPolicy
  }),
  false
);
assert.equal(
  canOwnProductionTileCompositorPresentation(viewerCanvasState),
  true
);

const isolatedCurrentTexturePath =
  await buildWebGpuViewerCanvasCurrentTexturePathReadiness({
    device: {},
    viewerCanvasState,
    canvasContextMutationAllowed: canMutateProductionPresentationState(
      diagnosticOwnershipPolicy
    )
  });
assert.equal(isolatedCurrentTexturePath.status, 'ok');
assert.equal(isolatedCurrentTexturePath.viewerCanvasContextMutationSuppressed, true);
assert.equal(isolatedCurrentTexturePath.viewerCanvasContextConfigured, false);
assert.equal(isolatedCurrentTexturePath.viewerCanvasCurrentTextureAcquired, false);
assert.equal(canvas.getContextCalls, 0);

const nonproductionCanvasCounters = {
  getContext: 0,
  configure: 0,
  getCurrentTexture: 0
};
const nonproductionCanvas = {
  width: 64,
  height: 64,
  getContext() {
    nonproductionCanvasCounters.getContext += 1;
    return {
      configure() {
        nonproductionCanvasCounters.configure += 1;
      },
      getCurrentTexture() {
        nonproductionCanvasCounters.getCurrentTexture += 1;
        return {};
      }
    };
  }
};
const nonproductionCurrentTexturePath =
  await buildWebGpuViewerCanvasCurrentTexturePathReadiness({
    device: {},
    viewerCanvasState: { ...viewerCanvasState, canvas: nonproductionCanvas }
  });
assert.equal(nonproductionCurrentTexturePath.viewerCanvasCurrentTexturePathReady, true);
assert.deepEqual(nonproductionCanvasCounters, {
  getContext: 1,
  configure: 1,
  getCurrentTexture: 1
});

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
