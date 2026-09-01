import assert from 'node:assert/strict';

import {
  buildWebGpu4DStatePositionsForCandidates
} from '../demo/js/webgpu_4d_state_evaluator.js';

const TOLERANCE = 1e-5;

// Inputs come from the Step119 WebGPU lineage artifact. Expected covariance
// values come independently from the same-index CUDA direct rasterizer artifact.
const fixtures = [
  {
    srcIndex: 658947,
    scale: [0.18787826597690582, 0.6774884462356567, 0.14989617466926575],
    scaleT: 0.053418248891830444,
    qL: [-0.412754625082016, -0.6678161025047302, 0.6754933595657349, -0.044882114976644516],
    qR: [0.9843918681144714, -0.09477174282073975, 0.15642258524894714, 0.049732476472854614],
    timeDelta: 0.4625465393066399,
    expectedTemporalDelta: [0.326629638671875, 0.20631790161132812, 0.05245041847229004],
    expectedCovariance: [0.038666367530822754, 0.00021630525588989258, 0.00036383606493473053, 0.020013142377138138, 0.015701526775956154, 0.01749798282980919]
  },
  {
    srcIndex: 771007,
    scale: [0.414957195520401, 0.21098582446575165, 0.0988846942782402],
    scaleT: 0.05628792569041252,
    qL: [-0.12781232595443726, -0.2942648231983185, -0.8474960327148438, 0.4437935948371887],
    qR: [0.9178687334060669, -0.23054656386375427, 0.03996846079826355, 0.07919143885374069],
    timeDelta: 0.45823211669921804,
    expectedTemporalDelta: [-0.0489044189453125, -0.15812301635742188, 0.27303361892700195],
    expectedCovariance: [0.012019973248243332, 0.0003195378230884671, 0.024793686345219612, 0.009862586855888367, -0.020455338060855865, 0.14096657931804657]
  },
  {
    srcIndex: 788034,
    scale: [0.09215150773525238, 0.18129105865955353, 0.5472375750541687],
    scaleT: 0.05572381243109703,
    qL: [-0.46359783411026, 0.027141714468598366, -0.3118325471878052, -0.9322201609611511],
    qR: [0.9005423188209534, -0.22483910620212555, 0.00986932311207056, 0.07999799400568008],
    timeDelta: 0.07670516967773366,
    expectedTemporalDelta: [0.09087371826171875, -0.18391036987304688, -0.06210160255432129],
    expectedCovariance: [0.06213017925620079, -0.08825606852769852, -0.03742782771587372, 0.13759168982505798, 0.052124664187431335, 0.05724497511982918]
  },
  {
    srcIndex: 826401,
    scale: [0.11786039918661118, 0.6684491634368896, 0.5093389749526978],
    scaleT: 0.0633649230003357,
    qL: [0.26853954792022705, -0.1493859440088272, 0.01680520363152027, -1.1372560262680054],
    qR: [0.8761600852012634, -0.06009817495942116, 0.23205120861530304, 0.061852630227804184],
    timeDelta: 0.16261215209960866,
    expectedTemporalDelta: [0.05800056457519531, 0.2542076110839844, -0.17354893684387207],
    expectedCovariance: [0.06655314564704895, -0.053985074162483215, -0.142842635512352, 0.22192126512527466, 0.07034882158041, 0.34726542234420776]
  },
  {
    srcIndex: 835183,
    scale: [0.35051074624061584, 0.07003045082092285, 0.12375552207231522],
    scaleT: 0.06619875878095627,
    qL: [0.2737041115760803, -0.9061931371688843, -0.3144143223762512, -0.3822574019432068],
    qR: [0.6737524271011353, 0.10580801963806152, -0.20505018532276154, -0.35926496982574463],
    timeDelta: -0.1177623748779304,
    expectedTemporalDelta: [-0.01590728759765625, 0.1028900146484375, 0.011130332946777344],
    expectedCovariance: [0.029255494475364685, 0.04405159130692482, 0.00003891540109179914, 0.09437839686870575, 0.0017052973853424191, 0.004714047536253929]
  },
  {
    srcIndex: 852955,
    scale: [0.12488022446632385, 0.11323174089193344, 0.5365891456604004],
    scaleT: 0.04891936480998993,
    qL: [0.2837536931037903, -0.27144649624824524, -0.8573729991912842, 0.6489059925079346],
    qR: [0.8598641753196716, -0.07317347824573517, -0.13328565657138824, 0.15259809792041779],
    timeDelta: -0.3944824218750007,
    expectedTemporalDelta: [-0.6056632995605469, -0.7036857604980469, -0.07423949241638184],
    expectedCovariance: [0.06786307692527771, 0.05329929292201996, 0.0013820650056004524, 0.047295644879341125, -0.0014791181311011314, 0.01420795451849699]
  },
  {
    srcIndex: 863505,
    scale: [0.4227825105190277, 0.09590434283018112, 0.112580306828022],
    scaleT: 0.06091422215104103,
    qL: [0.4172423481941223, 0.04269510507583618, -0.014289028011262417, 0.9991644620895386],
    qR: [1.0512104034423828, -0.015533972531557083, -0.018474489450454712, -0.15409812331199646],
    timeDelta: -0.28595046997070384,
    expectedTemporalDelta: [0.07155227661132812, -0.012969970703125, -0.010648250579833984],
    expectedCovariance: [0.003953133709728718, -0.00009755417704582214, -0.00021430745255202055, 0.01178068295121193, 0.0015585514483973384, 0.010136044584214687]
  },
  {
    srcIndex: 906711,
    scale: [0.1802777647972107, 0.2459844946861267, 0.5528331398963928],
    scaleT: 0.059968605637550354,
    qL: [-0.6082768440246582, 0.5573820471763611, -0.5009554028511047, -0.6815871596336365],
    qR: [1.0886996984481812, 0.018991656601428986, 0.10398376733064651, -0.069514200091362],
    timeDelta: -0.5074470520019538,
    expectedTemporalDelta: [0.3164329528808594, -0.2902545928955078, -0.39098167419433594],
    expectedCovariance: [0.0314304381608963, -0.033103734254837036, 0.010576948523521423, 0.0698196142911911, 0.013244040310382843, 0.03783869743347168]
  }
];

const dot = (a, b) => a.reduce((sum, value, index) => sum + value * b[index], 0);
const normalize = (values) => {
  const length = Math.sqrt(dot(values, values));
  return values.map((value) => value / Math.max(length, 1e-6));
};
const column = (rows, index) => rows.map((row) => row[index]);

function cudaConditionalCovarianceReference({ scale, scaleT, qL: qLRaw, qR: qRRaw }) {
  const [a, b, c, d] = normalize(qLRaw);
  const [p, q, r, s] = normalize(qRRaw);
  const ml = [
    [a, -b, c, -d],
    [b, a, -d, -c],
    [-c, d, a, -b],
    [d, c, b, a]
  ];
  const mr = [
    [p, -q, r, s],
    [q, p, -s, r],
    [-r, s, p, q],
    [-s, -r, -q, p]
  ];
  const rotationRows = mr.map((row) => ml.map((_, index) => dot(row, column(ml, index))));
  const rotationColumns = [0, 1, 2, 3].map((index) => column(rotationRows, index));
  const scaleSq = [...scale.map((value) => value * value), scaleT * scaleT];
  const sigma = (left, right) => dot(
    scaleSq.map((value, index) => value * left[index]),
    right
  );
  const spatialTemporal = [0, 1, 2].map((index) =>
    sigma(rotationColumns[index], rotationColumns[3])
  );
  const temporalVariance = Math.max(
    sigma(rotationColumns[3], rotationColumns[3]),
    1e-8
  );
  const spatialPairs = [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]];
  const conditional = spatialPairs.map(([left, right]) =>
    sigma(rotationColumns[left], rotationColumns[right]) -
      spatialTemporal[left] * spatialTemporal[right] / temporalVariance
  );
  return { conditional, spatialTemporal, temporalVariance };
}

function qLOnlyCovarianceReference({ scale, qL: qRaw }) {
  const [qr, qx, qy, qz] = normalize(qRaw);
  const rows = [
    [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qr * qz), 2 * (qx * qz + qr * qy)],
    [2 * (qx * qy + qr * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qr * qx)],
    [2 * (qx * qz - qr * qy), 2 * (qy * qz + qr * qx), 1 - 2 * (qx * qx + qy * qy)]
  ].map((row, index) => row.map((value) => value * scale[index]));
  return [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]].map(
    ([left, right]) => rows.reduce(
      (sum, row) => sum + row[left] * row[right],
      0
    )
  );
}

const maxComponentError = (actual, expected) => Math.max(
  ...actual.map((value, index) => Math.abs(value - expected[index]))
);

let maximumConditionalError = 0;
let minimumOldImplementationError = Number.POSITIVE_INFINITY;
let maximumOldImplementationError = 0;
let maximumTemporalError = 0;
for (const fixture of fixtures) {
  const reference = cudaConditionalCovarianceReference(fixture);
  const conditionalError = maxComponentError(
    reference.conditional,
    fixture.expectedCovariance
  );
  maximumConditionalError = Math.max(maximumConditionalError, conditionalError);
  assert.ok(
    conditionalError <= TOLERANCE,
    `srcIndex ${fixture.srcIndex} conditional covariance error ${conditionalError}`
  );

  const oldImplementationError = maxComponentError(
    qLOnlyCovarianceReference(fixture),
    fixture.expectedCovariance
  );
  minimumOldImplementationError = Math.min(
    minimumOldImplementationError,
    oldImplementationError
  );
  maximumOldImplementationError = Math.max(
    maximumOldImplementationError,
    oldImplementationError
  );
  assert.ok(
    oldImplementationError > TOLERANCE,
    `srcIndex ${fixture.srcIndex} must distinguish the old qL-only covariance`
  );

  const temporalDelta = reference.spatialTemporal.map(
    (value) => value * fixture.timeDelta / reference.temporalVariance
  );
  const temporalError = maxComponentError(
    temporalDelta,
    fixture.expectedTemporalDelta
  );
  maximumTemporalError = Math.max(maximumTemporalError, temporalError);
  assert.ok(
    temporalError <= TOLERANCE,
    `srcIndex ${fixture.srcIndex} temporal coupling error ${temporalError}`
  );
}

const previousGpuBufferUsage = globalThis.GPUBufferUsage;
globalThis.GPUBufferUsage = { STORAGE: 1, UNIFORM: 2, COPY_SRC: 4 };

let evaluatorShaderSource = null;
const createFakeBuffer = (descriptor) => {
  const mappedRange = new ArrayBuffer(descriptor.size);
  return {
    descriptor,
    getMappedRange: () => mappedRange,
    unmap: () => {},
    destroy: () => {}
  };
};
const fakePass = {
  setPipeline: () => {},
  setBindGroup: () => {},
  dispatchWorkgroups: () => {},
  end: () => {}
};
const fakeDevice = {
  limits: {
    maxStorageBufferBindingSize: 1_073_741_824,
    maxBufferSize: 1_073_741_824,
    maxStorageBuffersPerShaderStage: 8
  },
  createBuffer: createFakeBuffer,
  createShaderModule: ({ code }) => {
    evaluatorShaderSource = code;
    return { code };
  },
  createComputePipeline: () => ({
    getBindGroupLayout: () => ({ label: 'fake-evaluator-bind-group-layout' })
  }),
  createBindGroup: () => ({ label: 'fake-evaluator-bind-group' }),
  createCommandEncoder: () => ({
    beginComputePass: () => fakePass,
    finish: () => ({ label: 'fake-evaluator-command-buffer' })
  }),
  queue: {
    submit: () => {},
    onSubmittedWorkDone: async () => {}
  }
};

try {
  const result = await buildWebGpu4DStatePositionsForCandidates({
    device: fakeDevice,
    raw: {
      N: 1,
      activeShDegree: 2,
      activeShDegreeT: 2,
      t: new Float32Array([0]),
      tDim: 1,
      scale_t: new Float32Array([1]),
      scaleTDim: 1,
      f_dc: new Float32Array([0, 0, 0]),
      fdcDim: 3,
      f_rest: new Float32Array(45),
      frestDim: 45,
      scale_xyz: new Float32Array([1, 1, 1]),
      scaleXYZDim: 3,
      rotation: new Float32Array([1, 0, 0, 0]),
      rotationDim: 4,
      rotation_r: new Float32Array([1, 0, 0, 0]),
      rotationRDim: 4
    },
    candidateIndices: new Uint32Array([0]),
    rawXyzOpacity: new Float32Array([0, 0, 1, 0]),
    buildConfig: { timestamp: 0, scalingModifier: 1, sigmaScale: 1 },
    projectionParams: new Float32Array(24),
    readbackPolicy: 'none',
    keepGpuResources: true,
    sourceWorksetResourceIdentity: 'step120-impl1-smoke-workset'
  });

  assert.equal(typeof evaluatorShaderSource, 'string');
  assert.match(evaluatorShaderSource, /struct CudaConditional4dCovariance/);
  assert.match(evaluatorShaderSource, /fn cudaConditional4dCovariance\(/);
  assert.match(
    evaluatorShaderSource,
    /sigma00 - spatialTemporal\.x \* spatialTemporal\.x \* invCovT/
  );
  assert.match(
    evaluatorShaderSource,
    /let conditional4dCovariance = cudaConditional4dCovariance\([\s\S]*?qRaw,[\s\S]*?qRRaw[\s\S]*?\);/
  );
  assert.match(
    evaluatorShaderSource,
    /let temporalMean = cudaConditionalTemporalMeanOffset\([\s\S]*?dt,[\s\S]*?conditional4dCovariance[\s\S]*?\);/
  );
  assert.match(
    evaluatorShaderSource,
    /let cov00 = conditional4dCovariance\.spatial0\.x;[\s\S]*?let cov22 = conditional4dCovariance\.spatial1\.z;/
  );
  assert.doesNotMatch(evaluatorShaderSource, /let q = qRaw \/ max\(length\(qRaw\), 1e-6\);/);
  assert.match(
    evaluatorShaderSource,
    /footprintPayload\[outBase \+ 1u\] = vec4f\(cov00, cov01, cov02, cov11\);[\s\S]*?footprintPayload\[outBase \+ 2u\] = vec4f\(cov12, cov22, covCam00, covCam01\);/
  );

  const footprintContract = result.gaussianFootprintEvaluationContract;
  assert.ok(footprintContract.computedFootprintFields.includes('conditionalCovariance3D'));
  assert.equal(
    footprintContract.partialFootprintFields.includes(
      '4d-conditional-covariance-temporal-marginal-deferred'
    ),
    false
  );
  assert.equal(
    footprintContract.deferredFootprintFields.includes(
      'full-4d-conditional-covariance'
    ),
    false
  );
  assert.equal(footprintContract.fullGaussianFootprintEvaluationInWgsl, false);
  assert.equal(result.productionReadbackPerformed, false);

  for (const buffer of [
    result.gpuResources.statePositionBuffer,
    result.gpuResources.renderAttributeBuffer,
    result.gpuResources.footprintPayloadBuffer
  ]) {
    buffer.destroy();
  }
} finally {
  if (previousGpuBufferUsage === undefined) {
    delete globalThis.GPUBufferUsage;
  } else {
    globalThis.GPUBufferUsage = previousGpuBufferUsage;
  }
}

console.log('Step120 Impl1 conditional covariance smoke: OK', {
  fixtureCount: fixtures.length,
  tolerance: TOLERANCE,
  maximumConditionalError,
  minimumOldImplementationError,
  maximumOldImplementationError,
  maximumTemporalError
});
