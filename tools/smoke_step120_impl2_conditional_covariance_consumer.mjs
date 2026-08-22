import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

import {
  computeCudaConditionalGaussianState4D
} from '../demo/js/cuda_4d_state.js';

const TOLERANCE = 1e-5;

// Inputs are fixed from the Step119 WebGPU lineage artifact. Expected values
// are fixed independently from the same-index CUDA direct rasterizer artifact.
const fixtures = [
  {
    srcIndex: 658947,
    scale: [0.18787826597690582, 0.6774884462356567, 0.14989617466926575],
    scaleT: 0.053418248891830444,
    qL: [-0.412754625082016, -0.6678161025047302, 0.6754933595657349, -0.044882114976644516],
    qR: [0.9843918681144714, -0.09477174282073975, 0.15642258524894714, 0.049732476472854614],
    expected: [0.038666367530822754, 0.00021630525588989258, 0.00036383606493473053, 0.020013142377138138, 0.015701526775956154, 0.01749798282980919]
  },
  {
    srcIndex: 771007,
    scale: [0.414957195520401, 0.21098582446575165, 0.0988846942782402],
    scaleT: 0.05628792569041252,
    qL: [-0.12781232595443726, -0.2942648231983185, -0.8474960327148438, 0.4437935948371887],
    qR: [0.9178687334060669, -0.23054656386375427, 0.03996846079826355, 0.07919143885374069],
    expected: [0.012019973248243332, 0.0003195378230884671, 0.024793686345219612, 0.009862586855888367, -0.020455338060855865, 0.14096657931804657]
  },
  {
    srcIndex: 788034,
    scale: [0.09215150773525238, 0.18129105865955353, 0.5472375750541687],
    scaleT: 0.05572381243109703,
    qL: [-0.46359783411026, 0.027141714468598366, -0.3118325471878052, -0.9322201609611511],
    qR: [0.9005423188209534, -0.22483910620212555, 0.00986932311207056, 0.07999799400568008],
    expected: [0.06213017925620079, -0.08825606852769852, -0.03742782771587372, 0.13759168982505798, 0.052124664187431335, 0.05724497511982918]
  },
  {
    srcIndex: 826401,
    scale: [0.11786039918661118, 0.6684491634368896, 0.5093389749526978],
    scaleT: 0.0633649230003357,
    qL: [0.26853954792022705, -0.1493859440088272, 0.01680520363152027, -1.1372560262680054],
    qR: [0.8761600852012634, -0.06009817495942116, 0.23205120861530304, 0.061852630227804184],
    expected: [0.06655314564704895, -0.053985074162483215, -0.142842635512352, 0.22192126512527466, 0.07034882158041, 0.34726542234420776]
  },
  {
    srcIndex: 835183,
    scale: [0.35051074624061584, 0.07003045082092285, 0.12375552207231522],
    scaleT: 0.06619875878095627,
    qL: [0.2737041115760803, -0.9061931371688843, -0.3144143223762512, -0.3822574019432068],
    qR: [0.6737524271011353, 0.10580801963806152, -0.20505018532276154, -0.35926496982574463],
    expected: [0.029255494475364685, 0.04405159130692482, 0.00003891540109179914, 0.09437839686870575, 0.0017052973853424191, 0.004714047536253929]
  },
  {
    srcIndex: 852955,
    scale: [0.12488022446632385, 0.11323174089193344, 0.5365891456604004],
    scaleT: 0.04891936480998993,
    qL: [0.2837536931037903, -0.27144649624824524, -0.8573729991912842, 0.6489059925079346],
    qR: [0.8598641753196716, -0.07317347824573517, -0.13328565657138824, 0.15259809792041779],
    expected: [0.06786307692527771, 0.05329929292201996, 0.0013820650056004524, 0.047295644879341125, -0.0014791181311011314, 0.01420795451849699]
  },
  {
    srcIndex: 863505,
    scale: [0.4227825105190277, 0.09590434283018112, 0.112580306828022],
    scaleT: 0.06091422215104103,
    qL: [0.4172423481941223, 0.04269510507583618, -0.014289028011262417, 0.9991644620895386],
    qR: [1.0512104034423828, -0.015533972531557083, -0.018474489450454712, -0.15409812331199646],
    expected: [0.003953133709728718, -0.00009755417704582214, -0.00021430745255202055, 0.01178068295121193, 0.0015585514483973384, 0.010136044584214687]
  },
  {
    srcIndex: 906711,
    scale: [0.1802777647972107, 0.2459844946861267, 0.5528331398963928],
    scaleT: 0.059968605637550354,
    qL: [-0.6082768440246582, 0.5573820471763611, -0.5009554028511047, -0.6815871596336365],
    qR: [1.0886996984481812, 0.018991656601428986, 0.10398376733064651, -0.069514200091362],
    expected: [0.0314304381608963, -0.033103734254837036, 0.010576948523521423, 0.0698196142911911, 0.013244040310382843, 0.03783869743347168]
  }
];

const normalize = (values) => {
  const length = Math.hypot(...values) || 1;
  return values.map((value) => value / length);
};

function oldQlOnlyCovariance({ scale, qL }) {
  const [r, x, y, z] = normalize(qL);
  const rows = [
    [1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y)],
    [2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x)],
    [2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)]
  ].map((row, index) => row.map((value) => value * scale[index]));
  return [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]].map(
    ([left, right]) => rows.reduce(
      (sum, row) => sum + row[left] * row[right],
      0
    )
  );
}

function flattenConditionalCovariance(covariance) {
  return [
    covariance[0][0],
    covariance[0][1],
    covariance[0][2],
    covariance[1][1],
    covariance[1][2],
    covariance[2][2]
  ];
}

const maxComponentError = (actual, expected) => Math.max(
  ...actual.map((value, index) => Math.abs(value - expected[index]))
);

let maximumConditionalError = 0;
let minimumOldImplementationError = Number.POSITIVE_INFINITY;
let maximumOldImplementationError = 0;
for (const fixture of fixtures) {
  const state = computeCudaConditionalGaussianState4D({
    position: [0, 0, 0],
    opacity: 1,
    scaleXYZ: fixture.scale,
    scaleT: fixture.scaleT,
    rotation: fixture.qL,
    rotationR: fixture.qR,
    timestamp: 0,
    tCenter: 0,
    prefilterVar: -1
  });
  const conditionalError = maxComponentError(
    flattenConditionalCovariance(state.cov3),
    fixture.expected
  );
  maximumConditionalError = Math.max(maximumConditionalError, conditionalError);
  assert.ok(
    conditionalError <= TOLERANCE,
    `srcIndex ${fixture.srcIndex} conditional covariance error ${conditionalError}`
  );

  const oldImplementationError = maxComponentError(
    oldQlOnlyCovariance(fixture),
    fixture.expected
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
    `srcIndex ${fixture.srcIndex} must reject the old qL-only covariance`
  );
}

const runtimeSource = await readFile(
  new URL('../demo/js/webgpu_visible_record_dry_run_runtime.js', import.meta.url),
  'utf8'
);
const productionSource = await readFile(
  new URL('../demo/js/webgpu_4d_state_evaluator.js', import.meta.url),
  'utf8'
);

assert.match(
  runtimeSource,
  /function resolveCudaConditionalCovarianceInputsLocal\([\s\S]*?sourceScaleT \* scalingModifier \* sigmaScale[\s\S]*?rotationR[\s\S]*?return \{ scale, scaleT, rotation, rotationR \};/
);
assert.match(
  runtimeSource,
  /const sourceScaleY = finiteNumber\([\s\S]*?sourceScaleX[\s\S]*?\);[\s\S]*?const sourceScaleZ = finiteNumber\([\s\S]*?sourceScaleX[\s\S]*?\);/
);
assert.match(
  runtimeSource,
  /Math\.max\(Math\.max\(value, 1e-6\) \* scalingModifier, 1e-6\)/
);
assert.match(
  runtimeSource,
  /const conditionalState = computeCudaConditionalGaussianState4D\(\{[\s\S]*?scaleXYZ: scale,[\s\S]*?scaleT,[\s\S]*?rotation,[\s\S]*?rotationR,[\s\S]*?\}\);/
);
assert.doesNotMatch(runtimeSource, /function buildCudaRotationRowsLocal\(/);
assert.match(
  runtimeSource,
  /const covWorld = \[[\s\S]*?conditionalCovariance\[0\]\[0\],[\s\S]*?conditionalCovariance\[2\]\[2\][\s\S]*?\];[\s\S]*?cov3BilinearLocal\(view\[0\], view\[0\], covWorld\)/
);
assert.match(
  runtimeSource,
  /\? 'conditional-4d-to-3d-covariance'[\s\S]*?: maxCameraCovarianceAbsError/
);
assert.doesNotMatch(
  runtimeSource,
  /conditional4DCovarianceClassification: 'deferred'/
);
assert.doesNotMatch(
  runtimeSource,
  /webgpuPath: 'deferred; Step113 closes 3D rotation\/camera-Jacobian gap first'/
);
assert.match(
  runtimeSource,
  /expectedEvidenceSource:\s*'cuda-forward-cu-computeCov3D-conditional-computeCov2D-reference-formula'/
);

assert.match(runtimeSource, /const base = slot \* 32;/);
assert.match(
  runtimeSource,
  /covarianceWorldBeforeCameraTransform: \[[\s\S]*?readback\[base \+ 4\][\s\S]*?readback\[base \+ 9\][\s\S]*?\]/
);
assert.match(
  runtimeSource,
  /actualEvidenceSource:\s*'wgsl-step113-intermediate-diagnostic-readback-buffer'/
);
assert.match(productionSource, /const STEP113_DIAGNOSTIC_VEC4_STRIDE: u32 = 8u;/);
assert.match(
  productionSource,
  /footprintPayload\[outBase \+ 1u\] = vec4f\(cov00, cov01, cov02, cov11\);[\s\S]*?footprintPayload\[outBase \+ 2u\] = vec4f\(cov12, cov22, covCam00, covCam01\);/
);

console.log('Step120 Impl2 conditional covariance consumer smoke: OK', {
  fixtureCount: fixtures.length,
  tolerance: TOLERANCE,
  maximumConditionalError,
  minimumOldImplementationError,
  maximumOldImplementationError
});
