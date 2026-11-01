// One reviewed fixed scope; machine-inherited from the saved accepted candidate.
export const NORMAL_SCOPE_IDENTITY = 'b39024e0905c958093b75a11437e276520631edb33ef2805cc2699818fd3ae2d';
// This data does not authorize browser execution or claim normal-path acceptance.
function freeze(value) {
  if (value && typeof value === 'object') { Object.values(value).forEach(freeze); Object.freeze(value); }
  return value;
}
export const NORMAL_ADOPTED_SCOPE = freeze({
  "schema": "adopted-normal-scope-v1",
  "id": "adopted-b-fixed-range",
  "input": {
    "authority": {
      "index": {
        "sha256": "11263acf914d1478523b86c12c6e3337db23c511b1b1dbbf1cb2d52ebd18e028",
        "size": 18882
      },
      "manifest": {
        "sha256": "c2523cb7766395a727b91f2581345dd01b9c0e3cb8a17fb3e22ec0d40f7d269f",
        "size": 5198124
      },
      "receipt": {
        "sha256": "a28ccbdcc03a0d7a0e37684f82f07c74f06337dbfb3a136b0c9916f4377520ed",
        "size": 303
      },
      "result": {
        "sha256": "5f6aa8441cf6014a56c2566ef9c1aea595210a04669dcada48dfb7e23b0eebc4",
        "size": 581
      },
      "spl4": {
        "sha256": "be7270440616cc97707abff7e74856d92f0e56f528a16cc4486c854aa2f272f6",
        "size": 844714388
      },
      "verification": {
        "sha256": "af14ebb040e633f3af7dcec4e9dda3b19214f1d98027cdc78d58b4828c7f36dd",
        "size": 2811
      }
    },
    "baseUrl": "/step21-validation3/publication/",
    "selection": {
      "count": 65536,
      "representatives": [
        524288,
        589823,
        524324,
        572419,
        589817,
        529553,
        564754
      ],
      "start": 524288,
      "view": 4
    }
  },
  "rows": 3248901,
  "frame": [
    "train",
    2588,
    "images/000118_v16.png",
    16.6
  ],
  "mathVersion": "cuda-aligned-composite-v1",
  "depthOrder": "positive-depth-ascending-source-index-ascending-v2",
  "sortAlgorithm": "bounded-global-bitonic-stage-v2-depth-ascending-source-index",
  "authorization": {
    "instruction": "/home/demo/work/reports/webgpu-4dgs-viewer/phase3/step122/step122-normal-on-implement1/step122-normal-on-implement1-instruction.txt",
    "identity": {
      "size": 18317,
      "sha256": "b0959da95962925e0f6ebe49a704041b387882595dde696c181bd0b9abe2073e"
    },
    "connectionImplementationAuthorized": true,
    "browserExecutionAuthorized": false,
    "normalOnAccepted": false
  }
});
