export const NATIVE_WEBGPU_PRODUCTION_TILE_INPUT_BINDING_CONTRACT_VERSION =
  'phase3-native-webgpu-production-tile-input-bindings-v1';

const BINDING_DEFINITIONS = Object.freeze([
  Object.freeze({
    resourceKey: 'statePositions',
    binding: 0,
    wgslDeclaration: 'var<storage, read> statePositions: array<vec4f>;',
    layoutBufferType: 'read-only-storage'
  }),
  Object.freeze({
    resourceKey: 'renderAttributes',
    binding: 1,
    wgslDeclaration: 'var<storage, read> renderAttributes: array<vec4f>;',
    layoutBufferType: 'read-only-storage'
  }),
  Object.freeze({
    resourceKey: 'footprintPayload',
    binding: 2,
    wgslDeclaration: 'var<storage, read> footprintPayload: array<vec4f>;',
    layoutBufferType: 'read-only-storage'
  }),
  Object.freeze({
    resourceKey: 'projectionParams',
    binding: 4,
    wgslDeclaration: 'var<storage, read> projectionParams: array<vec4f>;',
    layoutBufferType: 'read-only-storage'
  }),
  Object.freeze({
    resourceKey: 'tileInputs',
    binding: 5,
    wgslDeclaration: 'var<storage, read_write> tileInputs: array<vec4f>;',
    layoutBufferType: 'storage'
  }),
  Object.freeze({
    resourceKey: 'params',
    binding: 6,
    wgslDeclaration: 'var<uniform> params: Params;',
    layoutBufferType: 'uniform'
  })
]);

export const NATIVE_WEBGPU_PRODUCTION_TILE_INPUT_BINDINGS =
  BINDING_DEFINITIONS;

function bindingNumbers(entries) {
  return Array.isArray(entries)
    ? entries.map((entry) => Number(entry?.binding))
    : [];
}

function duplicateBindings(bindings) {
  const seen = new Set();
  const duplicates = new Set();
  for (const binding of bindings) {
    if (seen.has(binding)) duplicates.add(binding);
    seen.add(binding);
  }
  return [...duplicates].sort((a, b) => a - b);
}

function compareBindingSets(actualBindings, expectedBindings) {
  const actual = new Set(actualBindings);
  const expected = new Set(expectedBindings);
  return {
    missing: expectedBindings.filter((binding) => !actual.has(binding)),
    extra: actualBindings.filter((binding) => !expected.has(binding))
  };
}

export function buildNativeWebGpuProductionTileInputWgslBindings() {
  return BINDING_DEFINITIONS.map(
    ({ binding, wgslDeclaration }) =>
      `@group(0) @binding(${binding}) ${wgslDeclaration}`
  ).join('\n');
}

export function buildNativeWebGpuProductionTileInputBindGroupLayoutEntries({
  computeVisibility
} = {}) {
  if (computeVisibility === undefined || computeVisibility === null) {
    throw new TypeError(
      'computeVisibility is required for the tile-input binding layout'
    );
  }
  return BINDING_DEFINITIONS.map(({ binding, layoutBufferType }) => ({
    binding,
    visibility: computeVisibility,
    buffer: { type: layoutBufferType }
  }));
}

export function buildNativeWebGpuProductionTileInputBindGroupEntries(
  resources = {}
) {
  return BINDING_DEFINITIONS.map(({ binding, resourceKey }) => {
    const buffer = resources?.[resourceKey] ?? null;
    if (!buffer) {
      throw new TypeError(
        `Missing production tile-input binding resource: ${resourceKey}`
      );
    }
    return { binding, resource: { buffer } };
  });
}

export function validateNativeWebGpuProductionTileInputBindingContract({
  shaderSource = '',
  layoutEntries = [],
  bindGroupEntries = []
} = {}) {
  const expectedBindings = BINDING_DEFINITIONS.map(({ binding }) => binding);
  const shaderBindings = [...String(shaderSource).matchAll(
    /@group\(0\)\s*@binding\((\d+)\)/g
  )].map((match) => Number(match[1]));
  const layoutBindings = bindingNumbers(layoutEntries);
  const descriptorBindings = bindingNumbers(bindGroupEntries);
  const shaderDuplicates = duplicateBindings(shaderBindings);
  const layoutDuplicates = duplicateBindings(layoutBindings);
  const descriptorDuplicates = duplicateBindings(descriptorBindings);
  const shaderSet = compareBindingSets(shaderBindings, expectedBindings);
  const layoutSet = compareBindingSets(layoutBindings, expectedBindings);
  const descriptorSet = compareBindingSets(descriptorBindings, expectedBindings);
  const shaderDefinitionMismatches = BINDING_DEFINITIONS.filter(
    ({ binding, wgslDeclaration }) =>
      !String(shaderSource).includes(
        `@group(0) @binding(${binding}) ${wgslDeclaration}`
      )
  ).map(({ resourceKey }) => resourceKey);
  const layoutDefinitionMismatches = BINDING_DEFINITIONS.filter(
    ({ binding, layoutBufferType }) => {
      const entry = layoutEntries.find(
        (candidate) => candidate?.binding === binding
      );
      return entry?.buffer?.type !== layoutBufferType;
    }
  ).map(({ resourceKey }) => resourceKey);
  const ready =
    shaderDuplicates.length === 0 &&
    layoutDuplicates.length === 0 &&
    descriptorDuplicates.length === 0 &&
    shaderSet.missing.length === 0 &&
    shaderSet.extra.length === 0 &&
    layoutSet.missing.length === 0 &&
    layoutSet.extra.length === 0 &&
    descriptorSet.missing.length === 0 &&
    descriptorSet.extra.length === 0 &&
    shaderDefinitionMismatches.length === 0 &&
    layoutDefinitionMismatches.length === 0;
  return {
    contractVersion:
      NATIVE_WEBGPU_PRODUCTION_TILE_INPUT_BINDING_CONTRACT_VERSION,
    ready,
    expectedBindings,
    shaderBindings,
    layoutBindings,
    descriptorBindings,
    duplicateBindings: {
      shader: shaderDuplicates,
      layout: layoutDuplicates,
      descriptor: descriptorDuplicates
    },
    missingBindings: {
      shader: shaderSet.missing,
      layout: layoutSet.missing,
      descriptor: descriptorSet.missing
    },
    extraBindings: {
      shader: shaderSet.extra,
      layout: layoutSet.extra,
      descriptor: descriptorSet.extra
    },
    shaderDefinitionMismatches,
    layoutDefinitionMismatches
  };
}
