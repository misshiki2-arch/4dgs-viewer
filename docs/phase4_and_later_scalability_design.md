# Phase 4以降のscalability・streaming・科学可視化設計

更新日: 2026-09-04 JST（外部一次資料確認）
状態: 将来設計候補。Viewerはcorrected training baselineとrestart gateの受入れまでfreeze中であり、本書はPhase 4開始、freeze解除、次Step、実装、外部方式の採用を認可しない。

## 1. この文書の目的

この文書は、次の共有チャットから、4DGS ViewerのPhase 4以降に
応用できる可能性がある設計要素を抽出し、現在のrepository設計と整合する形で
整理したものである。

- [4DGS Viewer 引継ぎ](https://chatgpt.com/share/6a86a655-5848-83e9-b5a8-815fd2be820c)
- [Casberry 3D調査結果](https://chatgpt.com/share/6a86a684-70f8-83e8-9c8c-e7b70e3e5db6)
- [Splat.js調査](https://chatgpt.com/share/6a8d2d56-faf4-83ee-8141-379ab0e249c7)

共有チャットは調査履歴と設計候補のsourceであり、そこに含まれるSNS投稿、
製品性能、未公開研究の主張を、そのまま採用済み要件とは扱わない。採用前には
公式仕様、paper、repository、license、再現benchmarkを個別に確認する。

現在地の正本は `docs/phase3_current_state.md`、Phase 3の長期設計は
`docs/phase3_webgpu_backend_design.md` である。この文書はそれらを上書きしない。

本書のPhase 4は、このViewer project内のscalability、streaming、GPU resource、
compression、LODだけに属する。別projectのPhase番号、ticket、設計はViewerの
現在地や実装認可へ持ち込まない。以下の外部方式も、read-only investigationまたは
将来のdesign spike候補であり、採用済みcontractではない。

## 2. 共有チャットから確認した固定方針

### 2.1 Viewerの責務

4DGS Viewerは、入力元に依存しない汎用的なブラウザ表示基盤である。

- WebGPUがproduction renderingを所有する。
- WebGL2はfallback、regression、comparison oracleとして残す。
- Three.jsはUI、camera、viewer shellのadapterであり、4DGS renderer semanticsの
  ownerにはしない。
- CUDAはreference、先行手法の評価、将来のオフライン変換に利用できるが、
  WebGPU Viewerの表示責務をCUDAへ移さない。
- SPH / VTUから4D Gaussianを直接生成する研究は、Viewer内部ではなく別の
  offline converter / research pipelineとして扱う。

### 2.2 Phase 4へ持ち込む対象

Phase 4以降で優先して検討する対象は、次の通りである。

- 大規模sceneのchunkingとout-of-core処理
- network streamingとprogressive loading
- GPU residency、cache、eviction、prefetch
- 空間LODと時間LOD
- payload compressionとformat adapter
- WebGPU向けsort、culling、overdraw削減
- device、GPU memory、network bandwidthに応じたbudget制御
- CUDA / exact WebGPU結果に対する性能・品質・物理量誤差評価

ゲーム物理、衝突判定、AIによる未観測領域生成、アバター、影、一般的な
image-to-3D生成は、Viewerのscalability責務とは別である。

## 3. 最重要のPhase境界

### 3.1 Phase 3終了前に必要なexact full-scene gate

現在のproduction resident worksetは、一つの連続rangeを選択し、device capacityを
超えるrecordを明示的なnon-resident recordとして扱う。これは正しいPhase 3の
bounded ownershipだが、scene全体を処理したことにはならない。

現在の3,231,588 source recordsについて、最終的なvisual parityまたはproduction
acceptanceを主張する前に、次のいずれかが必要である。

1. device limit内で全recordを同時residentにする。
2. correctness-preservingなchunk処理により、全recordを欠落なく処理する。

このexact chunkingは、Phase 4の近似LODではない。device limitを越えるsceneに
対してPhase 3 correctnessを成立させるために必要なら、Phase 3終了ゲートとして
先に実装する。

exact chunkingで複数chunkを処理するとき、各chunkを独立に最終画像へalpha合成
するだけでは、一般にはCUDAと同じ結果にならない。異なるchunkのGaussian間にも
depth orderがあるため、少なくとも次のいずれかを満たす必要がある。

- 全chunkが生成したtile referenceを統合し、同一のglobal per-tile orderingと
  compositorへ渡す。
- chunk間のdepth intervalが非交差で、front-to-backの合成順序を証明できる。
- 別方式を使う場合は、CUDA full-scene結果と同値であることをartifactで証明する。

silent omission、chunk単位の不正な合成、fixed resident rangeだけによる
full-scene完了宣言は禁止する。

### 3.2 Phase 4のapproximate scalability

Phase 4では、accepted parity boundaryの内側でperformanceとscalabilityを進める。
LOD、opacity-aware culling、early termination、quantizationなど、結果を近似する
可能性がある機能は、exact full-scene modeと別のacceptance contractを持つ。

したがってruntime modeは、少なくとも概念上、次を区別する。

| mode | 目的 | 完了条件 |
|---|---|---|
| exact full-scene | 全source populationの正しさを証明する | silent omissionなし、CUDA full-scene比較合格 |
| progressive exact | 表示途中は不完全でも、最終的にexactへ収束する | missing / inflightを明示し、収束後にexact gate合格 |
| budgeted LOD | device / network budget内で対話性能を得る | 近似内容、品質誤差、物理量誤差、poppingを別評価 |

## 4. 推奨するdata-plane architecture

共有チャットで比較されたSOG、Streamed SOG、RAD、SPZ、LCC2、3D Tiles等は、
一つの拡張子で優劣を決めるのではなく、次の四軸に分解する。

1. hierarchy: 空間と時間をどのように階層化するか。
2. payload: Gaussian属性をどのprecision、layout、codecで格納するか。
3. fetch: network / storageから何を一回の取得単位にするか。
4. semantics: source identity、time、物理属性、誤差情報をどう保持するか。

Viewer内部の推奨境界は次である。

```text
format-specific index / payload
-> decoder adapter
-> canonical scene + chunk contract
-> camera/time demand estimator
-> scheduler
-> fetch/cache
-> GPU residency owner
-> existing 4D state / footprint / tile / sort / compositor
```

rendererがSOG、SPZ、RAD、LCC等の形式名で分岐しないようにする。形式固有処理は
decoder adapterまでに閉じ、scheduler以降は共通contractを消費する。

Pascal Editorから参考にするのはrenderer codeそのものではなく、core state、
viewer、UI/editorを分け、dirtyな対象だけを更新する責務分離である。現在の
`common / webgpu / webgl2 / three` 境界を崩さず、streaming UIやload statusを
production mathへ混ぜない。

## 5. canonical chunk contract候補

将来のchunk contractは、少なくとも次の情報を持つ。

### Scene manifest

- schema / contract version
- canonical source formatとsource record count
- spatial boundsとtime range
- coordinate system、unit、orientation
- available attributesと各attributeのprecision
- hierarchy rootとchunk index location
- codec、checksum、byte rangeまたはURL
- exact / progressive / LODの対応mode
- dataset全体のsource identity

### Chunk descriptor

- stable chunk identity
- spatial node、temporal window、LOD level
- source index rangeまたは明示的source index mapping
- record count、compressed bytes、decoded bytes、GPU bytes estimate
- spatial bounds
- temporal bounds
- dynamic Gaussianのswept boundsまたは保守的4D bounds
- parent / child / neighbor relation
- payload checksumとdecode contract
- quantization scale、error bound、attribute omission policy
- persistent particle / Gaussian identityの有無

### Runtime residency state

- requested、queued、fetching、decoded、uploading、resident、evicting、failed
- missing record countとinflight record count
- last used frame / time
- GPU buffer identityとgeneration
- resident bytesとbudget contribution
- exact coverageかLOD coverageか
- fallback利用の有無

現在の `production-resident-workset-v1` が持つscene identity、original source
index space、resident row space、capacity、non-resident明示、fail-closed方針は、
この将来contractでも維持する。

## 6. 4D固有のhierarchy設計

静的3DGSの空間treeをそのまま採用するだけでは不十分である。4DGSではcameraに
加えてtimeが需要を決める。

推奨する選択keyは、概念上次の組である。

```text
(spatialNode, temporalWindow, lodLevel, attributeSet)
```

時間方向については、少なくとも次を検討する。

- 現在時刻を含むwindowを優先する。
- 再生方向の隣接windowをprefetchする。
- scrub時は古いprefetchをcancelまたは低優先度化する。
- window境界で同一recordが二重表示または欠落しないidentity規則を持つ。
- temporal LOD切替によるflicker、motion discontinuity、物理量jumpを評価する。

Gaussianが時間とともに空間nodeを移動する場合は、次の候補を比較する。

- reference timeの位置でpartitionし、swept boundsでvisibilityを保守的に判定する。
- temporal windowごとに空間partitionを作る。
- 4D boundsを用いてspatial / temporal selectionを同時に行う。

最初から複雑な4D treeを採用せず、実データのmotion、time count、fetch patternを
計測して選択する。

## 7. schedulerとGPU residency

schedulerの入力は次の通りとする。

- camera frustum、screen-space contribution、viewport
- current time、play direction、scrub velocity
- requested exactness / LOD quality
- device limits、GPU memory budget、record / tile-reference capacity
- network bandwidth、latency、cache状態
- chunk decode / upload cost
- attribute表示要求

schedulerの出力は、desired chunk setと優先順位であり、rendererへ直接recordを
注入しない。fetch/cacheとGPU residency ownerがgeneration付きresourceを作り、
既存production frame data pathへhandoffする。

evictionでは、camera距離だけでなく次を考慮する。

- 現在時刻と隣接時刻
- parent LODを残すか
- 再取得cost
- GPU tile-reference pressure
- last-valid presentationを壊さないこと
- exact modeでまだ未処理のchunkを欠落扱いにしないこと

load / evictのhysteresisを設け、cameraやtimeの小さな変化によるthrashingを防ぐ。

## 8. compressionと科学データの保護

共有チャットにあるSOG / Streamed SOG、SPZ、RAD、LCC系は、主にphotorealistic
display payloadを小さくする。4DGS Viewerがシミュレーション結果を扱う場合、
画像品質だけでcodecを選んではならない。

データを次の二層へ分ける。

### Render payload

- position、scale、rotation、opacity、color / SH、time parameters
- GPU uploadと描画を優先したlayout
- quantizationやcodecを許容できるが、error boundを記録する

### Scientific payload

- particle / source ID
- time identity
- velocity、pressure、density、vorticity等の物理属性
- coordinate / unit metadata
- original sourceへの対応

Scientific payloadは、用途に応じてlosslessまたは明示的error-boundedとする。
表示用SHを削減したことと、物理量を欠落・量子化したことを同じ扱いにしない。

SPL4 v2は現在、record-majorの全属性を一括parseする。将来形式では、SPL4の
semantic recordを直ちに捨てず、manifest / index / chunk payloadを外側へ加える
案を第一候補とする。既存SPL4をcanonical source、配信用payloadを派生artifactと
することで、圧縮codecの変更とrenderer semanticsを分離できる。

## 9. 参考にする技術と優先度

### 9.1 外部一次資料台帳

確認日はすべて2026-09-04 JSTである。paper、project page、code repository、
paper license、code licenseを別のsourceとして扱い、固定できないidentityは
「未確認」と記録する。

| 候補 | source / paper version | code availability | commit / release | license | Viewerで調査する責務 | 想定Stage | 現時点で証明していないこと |
|---|---|---|---|---|---|---|---|
| WebSplatter | [project page](https://websplatter.github.io/) / [arXiv:2602.03207v1](https://arxiv.org/abs/2602.03207)（2026-02-03） / [repository](https://github.com/websplatter/WebSplatter) | 公式repositoryを公開 | [`16bb06e86b8fb12193d4bb4cd2b0b3d4f3ab62be`](https://github.com/websplatter/WebSplatter/commit/16bb06e86b8fb12193d4bb4cd2b0b3d4f3ab62be)（2026-08-29）、releaseは未確認 | paper: arXiv non-exclusive distribution license、code: MIT | cross-device WebGPU execution、sort、rasterization、overdraw、temporary / peak memory | Stage 0、freeze解除後の明示的design spike | current parity、true 4D compatibility、device portability、local performance |
| KISS-GS | [project page](https://fraunhoferhhi.github.io/KISS-GS/) / [arXiv:2608.26948v1](https://arxiv.org/abs/2608.26948)（2026-08-27） / project-linked [ffsplat repository](https://github.com/w-m/ffsplat) | repositoryは公開。ただしpaper固有code releaseとの対応は未確認 | [`3fb6444c5faed6de9aa55fa2313df6c2a4104193`](https://github.com/w-m/ffsplat/commit/3fb6444c5faed6de9aa55fa2313df6c2a4104193)（2025-07-10）、paper対応releaseは未確認 | paper: arXiv non-exclusive distribution license、code: Apache-2.0 | post-training payload compression、decoder adapter、random access、GPU upload | Stage 0、将来のStage 5候補 | true 4D / time semantics、source identity、4D属性、error boundの保持 |
| SplatStream | [arXiv:2607.25971v2](https://arxiv.org/abs/2607.25971)（v1: 2026-07-28、v2: 2026-07-29） | arXivから公式project / repositoryを確認できず、code availabilityは未確認 | commit / release未確認 | paper: arXiv non-exclusive distribution license、code license未確認 | dynamic sceneのnetwork streaming、quality / resolution layer、temporal prediction、progressive packetization、bandwidth adaptation | Stage 0、将来のStage 4 / 5候補 | paper-level設計の実装可能性、final exact convergence、Viewer formatとの整合 |
| GPU-friendly Graphics Texture Coding | [arXiv:2607.14513v1](https://arxiv.org/abs/2607.14513)（2026-07-16） | arXivから公式project / repositoryを確認できず、code availabilityは未確認 | commit / release未確認 | paper: CC BY 4.0、code license未確認 | Gaussian attribute、特にSHのGPU向けcodec、random access、decode / upload | Stage 0、将来のStage 5候補 | WebGPU全deviceのformat対応、true 4D / time metadataの保持、local quality / performance |

paperが報告する数値性能は、本Viewerでの再現結果ではない。dataset、camera、time、
device、browser、driver、format、quality条件を固定して再測定するまで、設計採用条件や
本Viewerの性能値として扱わない。

### 9.2 採用前の共通評価軸

- true 4D attributeとtime semanticsへの適用可能性
- exact modeとの分離
- CUDAまたはaccepted WebGPU referenceに対する画質・semantic error
- device matrixとoptional WebGPU feature
- encoded bytes、decoded bytes、temporary memory、GPU resident bytes
- decode、upload、sort、raster、compositor時間
- random access、partial fetch、cache、evictionとの整合
- code licenseとformat specification
- deterministic behaviorおよびfail-closed behavior

### 9.3 優先度A: Phase 4 design spike対象

#### WebSplatter

[WebSplatter](https://github.com/websplatter/WebSplatter) はWebGPU nativeな
3DGS pipelineであり、cross-device WebGPU executionのdesign spike候補とする。
wait-free radix sort、dispatch間同期、hardware rasterizationの使い方、
opacity-aware処理、sort cost、temporary / peak memory、device portabilityを、
現在のglobal-storage bitonic orderingおよびrenderer責務と比較する。

現行sortまたはcompositorへ直ちに移植しない。accepted parity成立前のalgorithm
置換を認可せず、semantic mismatch修正とsort / rasterization変更を同じStepへ
混ぜない。freeze中は一次資料のread-only investigationに限定する。

#### KISS-GS

[KISS-GS](https://fraunhoferhhi.github.io/KISS-GS/) は、post-training payload
compressionおよびdecoder adapterの候補とする。pruning、attribute organization、
image-based encoding、decode cost、random access、GPU upload形式を、canonical
scene / chunk contractの外側に閉じられるか評価する。

static 3DGS向けの結果をtrue 4DGSへ自動一般化しない。source identity、time
semantics、4D属性、明示的error boundを保持できることを採用条件とし、paperと
project-linked repositoryの対応が固定できるまでは実装根拠にしない。

#### SplatStream

[SplatStream](https://arxiv.org/abs/2607.25971) は、dynamic Gaussian sceneの
network streaming候補として、quality / resolution layer、inter-layer / temporal
prediction、progressive packetization、bandwidth adaptationを調査する。

paper-level設計と公開実装の可能性を分ける。既存のexact full-scene、progressive
exact、budgeted LODへ直ちに統合せず、欠落を明示しながらfinal exact resultへ
収束できるかを独立に評価する。

#### GPU-friendly Graphics Texture Coding

[Compression of 3D Gaussian Splatting Data Using GPU-friendly Graphics Texture Coding](https://arxiv.org/abs/2607.14513)
は、Gaussian attribute、特にSHのGPU向けcodec候補とする。fixed-rate block
compression、random access、hardware decode、primitive reordering、bit-rate / quality
tradeoffを評価する。

BC1、BC7等をWebGPU全deviceで利用可能と仮定しない。browser feature、adapter、
device、format compatibilityを実測するまで採用せず、scientific属性、source
identity、time metadataへ同じlossy policyを自動適用しない。

#### Splat.js（arrival-space）

[arrival-space/splat.js](https://github.com/arrival-space/splat.js) は、従来の
表示用 `dylanebert/splat.js` とは別の、写真群からcamera poseを解き、3DGSを
学習し、標準PLYを出力するまでをbrowser内で行うMIT licensed実装である。
2026-08-25確認時の公式repository `main` は
[`2dd9e2de`](https://github.com/arrival-space/splat.js/commit/2dd9e2de12934fe73f0c901a73116165acad85c6)
である。更新が速い新規projectなので、調査時にはcommitを固定する。

Splat.jsのSfM / training機能をViewerへ取り込むことはPhase 4の目的ではない。
Viewerは既存3DGS / 4DGSの表示、Splat.jsは複数写真から静的3DGSを生成するpipelineで
あり、責務を分離する。一方、実際のbrowser WebGPU上で重いGaussian処理を成立
させている公開実装として、次をread-only design spikeの比較対象にする。

- Gaussian属性のGPU data layout、temporary buffer、peak memoryの管理
- global sorted binningと、現在のtile-reference / sort / compositor構成との差
- anisotropic Gaussian、SH、Mip-Splatting opacity compensationのshader境界
- Gaussian relocation / growthを含む可変populationのcapacity管理
- JavaScript WorkerによるCPU処理とWebGPU computeの分担
- 一つのWebGPU deviceをpipeline内で共有し、host-owned deviceも受け取れるownership API
- UIから独立したlibrary、薄いapp consumer、unit testとheadless-browser quality gateの分離
- PLYをinterchange境界にした生成pipelineとViewerの疎結合

公式READMEは、browser内SfM、WebGPU training、100万を超えるsplat、さらに200万・
400万Gaussianの公開exampleを記載している。ただし、これは本Viewerで再現した
device-independentなcapacity、VRAM、速度、品質の証拠ではない。特にGaussian数、
training時間、PSNR、対応browser、SH default等は更新され得るため、採用判断では
次を固定して測る。

- repository commit、browser / WebGPU implementation、GPU / driver
- dataset、入力画像数と解像度、training / evaluation split
- Gaussian数、SH degree、iteration / cycle、precision
- peak GPU memory、temporary memory、dispatch / sort時間
- PLY export後のattribute / coordinate / opacity semantics

Phase 4で参考にする優先順位は、まずbuffer ownership、sorting / binning、bounded
resource管理、test architectureである。SfM、Bundle Adjustment、training policy、
Gaussian growthそのものはViewer backendの対象外とする。Splat.jsがhost-owned deviceを
受け取れる事実も、production deviceとdiagnostic deviceを本Viewerで共有してよい根拠
にはしない。既存のdevice ownershipとfailure isolationを維持した上で比較する。

#### HiGS / gsplat experimental inference renderer

[gsplat](https://github.com/nerfstudio-project/gsplat) のexperimental inference pathは、
HiGS型のhierarchical tile設計を評価するCUDA reference候補である。Gaussianが
偏在するsceneでmacro-tile / render-tile分割が有効かをCUDAでbenchmarkし、効果が
確認できた考え方だけをWGSL向けに再設計する。

CUDA rendererをproduction Viewerにしない。

#### Streamed SOG / PlayCanvas

[Streamed SOG format specification](https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/streamed-sog/)
は、spatial tree、LOD別chunk、progressive loadingを明文化した参考実装である。
形式をそのままcanonical 4D formatにするのではなく、manifest、chunk descriptor、
global splat budget、progressive loadの設計を参考にする。

#### A LoD of Gaussians

[A LoD of Gaussians](https://felixwindisch.github.io/ALoDOfGaussians/) は、
external memoryを用いたout-of-core training / renderingの参考候補である。
Viewerではtraining部分ではなく、host memory、GPU memory、view-dependent working
set、large-scene residencyの境界を調査する。

#### Pascal Editor

[Pascal Editor](https://github.com/pascalorg/editor) は、core state、viewer、editor
UI、dirty updateの責務分離を確認する設計参考である。Gaussian rendererの数式や
低レベルWebGPU実装の参照とは扱わない。

### 9.4 優先度B: Phase 4後半以降

#### Predictive uncertainty / artifact-risk layer

[Predictive Photometric Uncertainty](https://arxiv.org/abs/2603.22786) の考え方は、
WebGPU画像が安定した後の評価拡張候補である。論文手法を直ちに実装するのではなく、
CUDA / exact WebGPU / WebGL2の差、projected anisotropy、overdraw、sort instability
等からartifact-riskを可視化する設計を先に検討する。

#### Format adapters

SOG、Streamed SOG、SPZ、glTF Gaussian extension等をdecoder候補として比較する。
open specification、license、browser decode cost、4D属性、source identity、
scientific payload保持を評価する。LCC / LCC2や閉じた形式は、仕様と合法的decoderが
確認できる場合だけinteroperability対象にする。

### 9.5 優先度C: 研究候補または現時点で非優先

- Softmax-GS等のcompositor変更は、Gaussian数削減の可能性があるが、現在の
  front-to-back alpha semanticsを変えるため、parity成立後の独立研究とする。
- textured splat、LGTM等はphotorealistic asset表示には有用だが、物理量保持の
  中核ではない。
- splat instancingは同一assetの大量再利用には有効だが、通常のSPH / VTU
  simulation outputでは重複assetが少ないため優先度を下げる。
- 3DGRT、影、屈折、反射はpresentation品質研究であり、scalability contractとは
  分離する。
- GaussianGPT、ArtiFixer、image-to-3D、単眼video-to-4DGSはcontent generation側で
  あり、Viewer backendのPhase 4責務ではない。
- Casberryはbrowser 3D UI / rapid authoringの動向としては興味深いが、4DGS
  scientific viewerのrenderer、format、streaming設計へ直接採用する要素は薄い。

## 10. acceptance contract

### 10.1 Exact chunk / streaming acceptance

- scene source record countとprocessed unique source countが一致する。
- duplicate、silent omission、out-of-bounds source indexがない。
- required / scattered / sorted / composited reference countが一致する。
- chunk境界を跨ぐdepth orderingがCUDA full-scene semanticsと一致する。
- fixed camera / timeでCUDA full-scene imageと所定のsemantic / visual toleranceを満たす。
- missing、inflight、failed chunkをculledと分類しない。
- observer readbackがproduction scheduling decisionにならない。

### 10.2 Progressive loading acceptance

- time-to-first-frameとtime-to-exact-convergenceを別に計測する。
- 初期表示が低LOD / partialであることをruntime contractに明示する。
- loadの進行が同一generationのlast-valid presentationを不正に破壊しない。
- fetch失敗、decode失敗、GPU upload失敗をblack / culledと混同しない。
- camera / timeを固定した場合、最終状態がexact resultへ収束する。

### 10.3 LOD acceptance

- exact full-scene resultをreferenceにする。
- image MAE / RMSEだけでなく、coverage、depth order、alpha、temporal flickerを測る。
- source / particle identityの保持率を測る。
- position、scale、time、表示中の物理量に対するerror boundを測る。
- LOD切替時のpoppingとtime再生時の不連続を測る。
- device budgetごとにresident records、resident bytes、frame time、fetch bytesを記録する。

### 10.4 Performance telemetry

最低限、次をstage別に記録する。

- manifest / index load time
- fetch bytes、request count、cache hit rate
- decode timeとtemporary CPU memory
- GPU upload bytes / time
- resident records / bytes、eviction count、thrash count
- 4D state、footprint、tile count / scatter、sort、compositor、presentation time
- visible records、tile references、overdraw proxy
- time-to-first-frame、time-to-target-quality、time-to-exact
- device limitsと選択されたbudget

一つのFPSだけで方式を採否判断しない。

## 11. 段階的な導入順

以下はfreeze解除後に別途認可を受ける場合の将来順序であり、いずれのStageも
本書によって開始済みにはならない。

### Stage 0: read-only investigation

- candidate repository / paper / licenseを確認する。
- 同一dataset、camera、time、deviceでbenchmark条件を固定する。
- WebSplatter、KISS-GS、SplatStream、GPU-friendly Graphics Texture Coding、
  Splat.js、HiGS、Streamed SOG、A LoD of Gaussiansから、algorithmとproduct claimを
  分離する。

### Stage 1: canonical manifest / chunk contract

- 既存SPL4を一つのchunkとして表現するcontractを先に作る。
- behaviorを変えず、scene identity、source index、time、attribute、capacityを
  manifest vocabularyへ移す。
- format decoderとrendererを分離する。

### Stage 2: exact multi-chunk full-scene path

- LODやquantizationを入れず、全record処理を成立させる。
- unified ordering / compositor semanticsを維持する。
- full-scene CUDA comparisonを通す。

このStageは、必要ならPhase 3 exit gateとして実施する。

### Stage 3: fetch / cache / residency scheduler

- local static chunkから開始する。
- network fetch、prefetch、evictionを一責務ずつ追加する。
- cameraとtimeの両方でdirty demandを生成する。

### Stage 4: progressive exact loading

- partial / inflight / exact-convergedを明示する。
- first-frame latencyを改善しつつ、最終結果のexactnessを維持する。

### Stage 5: budgeted LOD and compression

- spatial LODを先に評価する。
- temporal LODを別の責務として追加する。
- render payload compressionとscientific payload protectionを分ける。
- 近似ごとに独立したacceptance contractを作る。

### Stage 6: evaluation and scientific interaction

- uncertainty / artifact-risk表示
- physical attribute selection
- threshold、slice、probe、time comparison
- source / particle identity inspection

SPH direct converterの研究は、このViewer側Stageとは別計画で進め、canonical
format / chunk contractを接点にする。

## 12. 未決事項

- exact multi-chunk orderingを現在のtile-reference ownerへどう接続するか。
- current resource capacityで、record bufferとtile-reference bufferをどの単位で
  page化するか。
- dynamic Gaussianのspatial boundsをreference time、swept bounds、window別treeの
  どれで表現するか。
- time windowの長さとprefetch幅をどう決めるか。
- current SPL4 semanticsを維持したchunked containerを新設するか、既存open formatの
  extensionを採用するか。
- physical attributeをrender payloadと同時fetchするか、on-demand sidecarにするか。
- compression errorをimage errorとphysical quantity errorへどう配分するか。
- mobile / integrated GPU / discrete GPUでbudget policyをどう調整するか。

これらはPhase 3 semantic parityとfull-scene gateの結果、実datasetのmotion / time
distribution、device telemetryを確認してから決める。

## 13. 採用判断の原則

- SNS上の最大Gaussian数やFPSを、そのまま要件や再現結果にしない。
- CUDAで速い方式を、WebGPUへ直接移植できると仮定しない。
- 特定formatを採用する前に、hierarchy、payload、fetch、semanticsを分離して評価する。
- progressive displayとexact completionを分ける。
- LODとfull-population correctnessを分ける。
- photorealistic image qualityとscientific fidelityを分ける。
- Viewerとoffline SPH converterを分ける。
- 一つのStepでは、一つのruntime責務または一つの調査責務だけを扱う。
