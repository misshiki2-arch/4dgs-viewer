# Phase 4以降のscalability・streaming・科学可視化設計

更新日: 2026-10-30 JST（SOG v3提案のblind属性・公開状況を訂正し、Spark Radix・GaussianSplineを未採用候補として追加。Kopanas・既存10月文献を維持）
状態: 将来設計候補。現在の作業範囲とfreezeは[現在地正本§24](phase3_current_state.md#step122-adopted-range)に従う。本書はPhase 4開始、一般freeze解除、新Step、実装、外部方式の採用を認可しない。

## 1. この文書の目的

この文書は、次の共有チャットから、4DGS ViewerのPhase 4以降に
応用できる可能性がある設計要素を抽出し、現在のrepository設計と整合する形で
整理したものである。

- [4DGS Viewer 引継ぎ](https://chatgpt.com/share/6a86a655-5848-83e9-b5a8-815fd2be820c)
- [Casberry 3D調査結果](https://chatgpt.com/share/6a86a684-70f8-83e8-9c8c-e7b70e3e5db6)
- [Splat.js調査](https://chatgpt.com/share/6a8d2d56-faf4-83ee-8141-379ab0e249c7)

- [10月のGaussian Splatting研究・実装情報](https://chatgpt.com/share/6ae03407-6d90-83ec-b094-365f1265546d)

共有チャットは調査履歴と設計候補のsourceであり、そこに含まれるSNS投稿、
製品性能、未公開研究の主張を、そのまま採用済み要件とは扱わない。採用前には
公式仕様、paper、repository、license、再現benchmarkを個別に確認する。

現在地の正本は `docs/phase3_current_state.md`、Phase 3の長期設計は
`docs/phase3_webgpu_backend_design.md` である。この文書はそれらを上書きしない。

研究program全体の目的階層とprojectの役割は
[`50_4DGS_RESEARCH_PROGRAM_GOALS_JA.md`](../../4dgs-development-governance/50_4DGS_RESEARCH_PROGRAM_GOALS_JA.md)
を正本とする。Viewerは直接変換結果の利用・検証基盤であり、研究programの中心成果または
direct converterのownerではない。

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

direct converterとはcanonical format、manifest、provenance、粒子identity、時刻、物理属性等の
明示的interfaceで接続し、ViewerのPhase／Stepをconverterまたは研究programへ流用しない。

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

採用入力の全source recordsについて、最終的なvisual parityまたはproduction acceptanceを主張する前に、次のいずれかが必要である。現行の入力identityと点数は[現在地正本§24](phase3_current_state.md#step122-adopted-range)に従い、本節制定時の旧3,231,588件を固定値として継承しない。

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

LOD、streaming、compressionは現在の必須実装ではない。real-time利用が実測上成立しない場合に
限り、必要性と影響を独立責務として再評価する。以下の記述はその場合に備えた将来候補と
技術契約の記録であり、この再評価条件または別途の実装認可を置き換えない。

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

Issue #40の[Spark 2.2.0変更履歴](https://github.com/sparkjsdev/spark/blob/main/CHANGELOG.md)は、
scene変更後の旧scene chunkのfetch/decode継続と、不要な初回GPU uploadの修正を報告する。
ここからのViewerへの応用判断として、将来の評価時にはscene切替/disposeで不要処理を止めること、
停止不能な処理は古いgenerationの結果を破棄すること、再利用可能な常駐dataを不要に再転送しないことを
確認候補に加える。これは上流の修正内容とViewer側の評価条件を区別した補足であり、具体API、
buffer/schema、取消方式を決めない。既存のscrub時prefetch取消とresource handoffは変更しない。

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

次表の確認日は2026-09-04 JSTである。paper、project page、code repository、
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

<a id="issue-19-source-ledger"></a>

#### Issue #19 v2からの追加台帳（確認日: 2026-09-11 JST）

[受入済み報告v2](../../reports/webgpu-4dgs-viewer/issue-19/issue-19-investigation-report-v2.md)
第3節の固定版を登録する。release/commitと固定source URLは
[provenance-v2.json](../../reports/webgpu-4dgs-viewer/issue-19/provenance-v2.json)
から機械転記・照合した。特定paperの再現ではなく、paper versionは該当なしである。

| 対象 | 固定release / commit | code availability | license確認範囲 | Viewerでの候補責務 |
|---|---|---|---|---|
| SuperSplat | v3.0.0 / [`12398f7f6997bd59bdd82876fd90fc87a6ec6f68`](https://github.com/playcanvas/supersplat/tree/12398f7f6997bd59bdd82876fd90fc87a6ec6f68) | 公開source | [root LICENSE](https://github.com/playcanvas/supersplat/blob/12398f7f6997bd59bdd82876fd90fc87a6ec6f68/LICENSE): MIT | Phase 3 resource所有権・寿命・sourceRow点検、Phase 4 sort/indirect比較 |
| PlayCanvas Engine | v2.22.0 / [`b2c5d42bfb98e334b60cc1214c63c763046a07d7`](https://github.com/playcanvas/engine/tree/b2c5d42bfb98e334b60cc1214c63c763046a07d7) | 公開source | [root LICENSE](https://github.com/playcanvas/engine/blob/b2c5d42bfb98e334b60cc1214c63c763046a07d7/LICENSE): MIT | Phase 4 portable GPU radix/indirect比較 |
| fVDB-core | v0.5.1 / [`316639f1adea4f09cde41f61954c399538cc1b88`](https://github.com/openvdb/fvdb-core/tree/316639f1adea4f09cde41f61954c399538cc1b88) | 公開source | [root LICENSE](https://github.com/openvdb/fvdb-core/blob/316639f1adea4f09cde41f61954c399538cc1b88/LICENSE): Apache-2.0（読んだkernelのSPDXも確認） | Phase 3限定pixel寄与診断、Phase 4指定pixel/active tile診断 |

SuperSplat独自のProjectedSplatRendererと、そこから利用するEngine ComputeRadixSortを
根拠とする。Engineの一般的なhybrid rendererと同一構成とは扱わない。license確認は
上記rootと報告で読んだsourceの範囲であり、splat-transformやPyTorch/CUDA/CUB等の
依存全体・再配布条件は未監査である。移植・再配布の承認ではなく、将来利用時に表示保持、
適用NOTICE、変更表示等を確認する。記事の著者側測定やGTC sessionの紹介を本Viewerの
実測または固定kernelの説明へ読み替えない。根拠詳細は報告第3節を参照する。

<a id="issue-40-source-ledger"></a>

#### Issue #40の追加台帳（確認日: 2026-09-19 JST）

指定一次資料と公式version/licenseリンクだけを確認した。仕様、公開Issueの報告、論文abstractの
主張とViewerへの応用判断を区別する。以下は候補登録であり、未採用・未実装・本Viewerで未実測である。
既存台帳の固定版は変更せず、今回取得した現行ページを過去版のsource監査として扱わない。

| 候補 | 一次資料・paper版 | release / commit確認範囲 | license / code確認範囲 | 用途・未確認境界 |
|---|---|---|---|---|
| Three.js GaussianSplat | [公式仕様](https://threejs.org/docs/pages/GaussianSplat.html) / [公開Issue #34527](https://github.com/mrdoob/three.js/issues/34527)。paper該当なし | Issueはr186を明記。現行docsとr186実装の同一性・commitは未確認 | [r186 root LICENSE](https://github.com/mrdoob/three.js/blob/r186/LICENSE): MIT。依存全体は未監査 | Phase 3の補助比較。更新API不足は公開報告であり、source確認済み仕様としない |
| Gaussian Splat Lite | [開発者の1.0紹介](https://discourse.threejs.org/t/gaussian-splat-lite-1-0-webgpu-streaming-lod-and-depth-rendering-for-three-js/94234) / [公式repository](https://github.com/WilliamLiu-1997/Gaussian-Splat-Lite)。paper該当なし | [CHANGELOG](https://github.com/WilliamLiu-1997/Gaussian-Splat-Lite/blob/main/CHANGELOG.md)の1.0.0（09-09）と1.1.0（09-17）を区別。固定commit未確認 | 公開code。[root LICENSE](https://github.com/WilliamLiu-1997/Gaussian-Splat-Lite/blob/main/LICENSE): Apache-2.0。NOTICE・third-party全体は未監査 | Phase 3比較 / Phase 4 streaming・LOD・residency。1.1.0のfastSort既定有効は精度trade-offを伴うとの変更履歴であり、1.0の比較条件へ混ぜない |
| Spark | [CHANGELOG 2.2.0](https://github.com/sparkjsdev/spark/blob/main/CHANGELOG.md)。paper該当なし | 2.2.0（2026-09-11）の記載確認。固定commit未確認 | [main root LICENSE](https://github.com/sparkjsdev/spark/blob/main/LICENSE): MIT。2.2.0全code・Rust/依存のlicense対応は未監査 | Phase 4の取消・residency評価条件。上流修正は本Viewerの不具合・改善実績を意味しない |
| MoQSplat | [arXiv v1](https://arxiv.org/abs/2609.18624v1)（2026-09-16）、abstract / metadata | code公開先の案内あり。code release / commit未確認 | paper: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)。code license未確認 | 空間・品質階層と配信優先度。true 4D、exact convergence、Viewerでの効果は未確認 |
| DecoGS | [arXiv v1](https://arxiv.org/abs/2609.17230v1)（2026-09-15）、abstract / metadata | code availability / release / commit未確認 | paper: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。code license未確認 | 選択的最適化から更新量削減への着想。GPU buffer分離はViewer側の未検証応用 |
| PanoGS-SLAM | [arXiv v1](https://arxiv.org/abs/2609.17387v1)（2026-09-15）、abstract / metadata | abstractはcode公開予定と記載。release / commit未確認 | paper license・code license未確認 | パノラマ入力・姿勢推定の低優先・保留資料。投影一致の根拠にしない |

版・licenseの未確認を埋めるための追加探索や移植は行っていない。将来利用時の版固定・license確認は
別途必要であり、この台帳は導入・再配布の許諾判断ではない。Phase 3の比較条件は
[比較候補節](phase3_webgpu_backend_design.md#issue-40-comparison-candidates)を参照する。

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

<a id="issue-19-scalability-candidates"></a>

#### SuperSplat / Engine: portable GPU radixとindirect接続（未採用・未実測）

Issue #19 v2のS2/S3・P4-Aを、上記WebSplatter等と同じsort比較の候補として登録する。
用途は大量tile参照のsort時間、dispatch数、一時容量の改善可能性の評価である。現行には
GPU count/scan/scatter、compact tile span別global-storage bitonic、indirect compositorが
既にあり、追加価値はGPU化自体ではなくsort方式、key/payload配置、可変件数indirect接続にある。
portable 4-bit radixを比較基準候補とし、OneSweep 8-bitはsubgroups、lane、forward-progress等の
前提を確認できるGPU/backendに限る追加候補とする。20-bit depth量子化、8-bit alpha保存、
外部の投影・weight式やstochastic modeを一括採用しない。

別途比較を認可する場合は、同じtile参照入力とViewerのfull-precision key/tie契約を固定し、
欠落・余分な重複ゼロ、tile所属・depth/tie順序・payload ID・合成結果の一致を先に検証する。
同一Gaussianの異なるtileへの正当な参照は保持する。tile別radixとtile/depth結合key案は
key幅、pass数、tie保持、一時bytesで比較し、参照件数とtile偏りを変えてsort時間と総frame時間、
peak容量を測る。高速化、損益分岐規模、device portabilityは未確認であり、小規模での優位も
仮定しない。外部rendererの完成画像との一致は、方式活用の条件にしない。

#### fVDB: 指定pixel / active tileによる診断の大規模化（未採用・未実測）

Issue #19 v2のF1–F4・P4-Bを、
[Phase 3の限定pixel寄与診断候補](phase3_webgpu_backend_design.md#issue-19-diagnostic-candidates)
で必要性が確認された情報だけを大規模sceneでも取得する候補として登録する。指定pixelから
active tileを作り、診断の処理・出力量を問題領域へ限定する効果を期待する。既存productionの
active-tile dispatchとは用途を区別し、全画面高速化、LOD、圧縮、Gaussian streamingの解決としない。

将来の比較では同じ選択pixelのID/weightが全体版と一致することを確認し、input/camera/time、
source remap、sort/tie、閾値と打切り規則をViewerへ合わせる。fresh diagnostic分離とproduction
非依存を維持し、pixel数・最大寄与数・最大bytesを制限する。範囲外、top-k、予算超過は不完全な
取得として明示する。非top-kの一時ID/weight配列は最終ragged出力とは別に最大寄与数に依存し、
投影・交差・sort等の全体処理とhost同期も残り得るため、固定費を含む時間とpeak容量で評価する。
削減範囲、WGSL/device limitsへの適合、原因特定の有効性、実際の性能・容量効果は未確認である。

両候補の根拠詳細は[Issue #19 v2](../../reports/webgpu-4dgs-viewer/issue-19/issue-19-investigation-report-v2.md)
第5–8節を参照する。評価軸はViewerでの効用とし、別projectの進捗を不採用・優先順位の理由に
しない。既存freeze・accepted parity境界を維持し、この登録から追加調査、方式採用、実装、
benchmark、Phase/Step開始へ自動的に進まない。

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

- **低レベルGPU最適化の参考（未採用・本Viewerで未検証）:** George Kopanas,
  [What the 3DGS Literature Left on the Table: Five Self-Contained CUDA Optimizations for 3D Gaussian Splatting](https://grgkopanas.github.io/3dgs_cuda_blog/)。
  相談役確認日・CODEX再確認日はいずれも2026-10-29 JST。既存研究を統合した後のボトルネックと、
  改善／逆効果の理由を段階別に分析した資料である。Viewerでは実測された遅い箇所に応じ、
  Forwardの性能分析、メモリアクセス／レジスタ使用、分岐・cullingの考え方を参照する。
  A100/CUDAでの著者測定であり、WGSL／本Viewerへの移植性・速度・画像精度を保証しない。
  Backwardの勾配最適化は表示責務外で、記事のSH streamingも主な改善対象がBackwardのため現行SH0への直接の優先度は低い。
  失敗例も参考になるが、著者は途中版を保存しておらず、その差分時間を厳密に再現できないと注記する。
  固定code／license監査・独立再現は未実施。必要性はSection 3.2、採用前評価はSection 9.2／10に従い、
  手法の一括導入、既存候補の優先順位変更、現在の作業追加やCUDA学習高速化研究の開始にはしない。

#### Streamed SOG / PlayCanvas

[Streamed SOG format specification](https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/streamed-sog/)
は、spatial tree、LOD別chunk、progressive loadingを明文化した参考実装である。
形式をそのままcanonical 4D formatにするのではなく、manifest、chunk descriptor、
global splat budget、progressive loadの設計を参考にする。
形式versionと科学属性への接点は[未採用のSOG v3候補](#sog-v3-candidate)を参照する。

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

<a id="sog-v3-candidate"></a>

##### SOG v3補足（未採用・本Viewerで未実装／未実測）

相談役確認日・CODEX保存資料確認日: 2026-10-30 JST。取得本文・metadataとbytes identityは
[一次資料台帳](../../reports/webgpu-4dgs-viewer/literature-20261030-docsync1/literature-20261030-docsync1-sources.json)で固定する。
[公式SOG仕様](https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/sog/)はv2、
[Issue #362](https://github.com/playcanvas/splat-transform/issues/362)は2026-10-28公開のOpenなv3提案である。
[Engine PR #9873](https://github.com/playcanvas/engine/pull/9873)の公開・Open／未マージを確認した。
Issue本文はEngineのblind fields対応を後続予定とし、SplatTransformのPRリンクは今後としている。
参照実装完成は**作者報告**であり、提案schema・PRの存在・製品release・実動作／全属性対応の確認を同一視しない。
PR本文には旧blind列の説明も残るため、最新の複数field提案への対応完了とは扱わない。code diff監査・動作再現は未実施。
[公式repository](https://github.com/playcanvas/splat-transform)の所在確認と、v3実装・依存のlicense監査は別であり、
後者は未実施。既存台帳の別版の監査を流用しない。SplatTransform製品versionとSOG形式versionも区別する。

以下は提案内容の要点であり、採用契約ではない。

| 項目 | 提案上の変更 |
|---|---|
| 回転 | v2のsmallest-threeから24-bit exponential-map packingへ変更し、既存3画像の未使用alphaへ格納。quats.webpを除去 |
| opacity | codebook化し、1超の値も扱う案。描画契約への影響は下記で分離 |
| SH | degree 4（DC以外24係数/色、DC込み25）を追加。提案の「4 bands」を通常のband数と混同しない。palette labelは24-bitへ拡張。現行SH0比較の必要条件にはしない |
| blind | `meta.blind`に複数の`blind_<i>.webp`を列挙し、各画像のGaussian当たり4バイトへ名前・型・byte offset付きfieldを配置。型はu8/u16/u24/u32、f16/f32、unorm8/unorm16。総属性量を4バイトに限定しない |
| extras | 任意JSON。v2およびStreamed SOGにも追加する案 |

SOG全体の量子化は非可逆であり、lossless WebPが保持するのは量子化後のバイトである。
v2専用readerはv3を拒否するversion境界を維持する。提案上のv2出力ではopacityを1へ制限し、
blindを落とし、SHをdegree 3までに縮減するため、将来の互換性検査事項とする。

**Viewer側の検討案と未確認事項**は次のとおり。具体schema・converter algorithmは決定しない。

- blindを粒子ID／外部対応表への参照に使うほか、圧力・密度scalarや速度成分等を複数fieldへ格納する余地を
  Viewer側の利用案として評価する。型付き格納だけで単位、座標、粒子／時刻identity、複数成分の意味や
  時間系列を保証する汎用科学形式になるわけではない。具体schemaは決定しない。
- f16／unormの量子化誤差と、f32のIEEE bits格納・変換往復での保持実証を分ける。
  A byteを使う画像はalpha=0でもRGBを保持するlibwebp exactが必要であり、一般のlossless指定だけでは
  IDやfloat bitsの保持を保証しない。幅・名前空間・dataset/time対応、並べ替え／filter／chunk化／再出力を
  通したID完全一致と各fieldの保持を将来確認する。
- LoD統合Gaussianが複数粒子を代表する場合、単一粒子IDとの一対一対応を仮定せず、元粒子集合と
  集約方法と保存量・属性対応の追跡を検討する。物理量・単位・時刻・provenanceはSection 8のscientific payload側へ保持する。
  これはrender／scientific payloadの論理分離であり、必須sidecar化やSOGへのcanonical置換は決定しない。
- extrasの任意JSONは意味を自動保証しない。提案ではscene結合時に全入力のextrasが一致する場合だけ
  保持されるため、単位・出典等の必須情報を黙って失わないかを将来確認する。

opacity > 1は保存だけでなく、頂部が平らな分布とfootprint拡張を含む描画提案であり、fragmentの
ブレンドalphaをそのまま1超にする意味ではない。提案の「mass」はSPH物理質量・保存量保証ではない。
Section 4のdecoder／共通contract／renderer分離を維持するが、decoderだけで完結するとは限らず、
共通描画契約に収まるかは別途評価する。形式名によるrenderer分岐や数式採用は行わない。
現在のStep122 alpha cap 0.99・CUDA比較基準・残差調査とは別の将来候補である。

評価はSection 9.2／10へ接続し、SOG v2/v3・SPZ・現行canonical入力とその派生payloadを、
容量、decode/upload、CPU/GPU一時容量、復元誤差、ID完全一致、LoDによる意味変化で比較する。
具体bench・閾値・API・bufferは今回設計しない。canonical科学データと配信用派生データの分離、
true 4D時間属性の未解決部分、direct converterの別ownerを維持し、SPH直接変換をViewer Phase 5とは呼ばない。
上流の圧縮率・PSNR・回転誤差の数値を本Viewerの効果予測に転用しない。必要性はSection 3.2に従い、
候補登録だけで実装優先順位を変えず、Phase 4開始・形式採用・実装を認可しない。

### 9.5 優先度C: 研究候補または現時点で非優先

- **PanoGS-SLAM**は[追加台帳](#issue-40-source-ledger)に登録したパノラマ入力・姿勢推定の
  低優先・保留資料である。球面領域のSLAMという論文の対象をWebGPU/WebGL2投影一致の根拠や
  現在の必須camera要件へ昇格しない。
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

<a id="issue-40-scalability-candidates"></a>

### 9.6 Issue #40: streaming・residency・時間更新量の候補（未採用・未実測）

根拠の版と確認範囲は[追加台帳](#issue-40-source-ledger)へ集約する。以下は候補登録だけであり、
追加研究・実装・計測の開始ではない。Section 3.2のreal-time不成立時の再評価条件を維持する。

- **Gaussian Splat Lite**: 開発者説明と公式READMEが示すRAD/SOGのcamera-driven LOD、
  decode/cache/upload budget、読み込み中の旧detail保持・fadeを、将来のstreaming/GPU residency
  比較材料にする。これらの表示継続をexact coverageの証明にしない。4Dのtime需要、source identity、
  科学属性、欠落を明示したexact convergenceへの適用は未確認。Phase 3の描画経路比較は
  [別節](phase3_webgpu_backend_design.md#issue-40-comparison-candidates)とする。
- **Spark 2.2.0**: 変更履歴が示す旧sceneのfetch/decode停止と不要upload防止、実験的opt-inの
  LOD traversal改善を将来の比較材料にする。Viewer側への適用条件はSection 7、追加計測は
  Section 10.4へ集約する。上流の高速化率をViewerの予測値・受入閾値へ転用しない。
- **MoQSplat**: 論文abstractは空間Tracks、Groups、段階的品質Subgroupsと、視域・距離等に応じた
  配信を提案し、prototypeの部分評価を報告する。空間分割・品質階層・配信優先度の候補とするが、
  3D配信階層は4Dのtime windowではない。Section 4のcanonical dataとtransportを分離して参照する
  のはViewerへの応用判断であり、論文の実証成果ではない。MoQ/HTTP Range等の選定は未決のままとし、
  pruning・間引き等の近似品質向上と、全recordを欠落なく処理するexact convergenceを区別する。
- **DecoGS**: 論文abstractの動き・測光変化を持つ領域への選択的最適化は、時間方向の不要更新削減の
  着想となる。ただしViewerの静的/動的GPU buffer分離へ拡張する部分は未検証の応用案である。
  SPHのID・時刻・瞬間速度だけでは、位置・形状・色・alpha・物理属性の区間内不変性やGaussianとの
  対応を保証できない。表示側で安全な再利用を示す条件は未確認であり、学習・変換・分類algorithmの
  設計やdirect converterの責務へ広げない。

既存hierarchy/payload/fetch/semantics分離、schedulerからrendererへの責務境界とexact/approximate
分離を再設計しない。技術的効用はViewerで評価し、別projectの進捗による実施時期の制約と混同しない。

<a id="literature-20261027-candidates"></a>

### 9.7 2026-10-27文献補足：更新量・LOD・圧縮・別描画方式の候補

次は未採用・未実装・本Viewerで未実測の候補登録である。調査の入口は
[共有チャット](https://chatgpt.com/share/6ae03407-6d90-83ec-b094-365f1265546d)、
技術的根拠は下表の一次資料とする。共有チャットの添付原文・画像は取得できておらず、
それらを確認済み資料に含めない。共有チャット内の要約や性能値を、そのまま設計要件へ転記しない。

相談役による一次資料の確認日は2026-10-27 JST。今回の文書同期では、その確認範囲を登録する。paper versionは下表で固定する。project pageとIssueは同日の観測であり、
固定codeの監査ではない。以下の候補固有codeのrelease／commit、依存を含むcode licenseは今回未確認とする。
paper licenseはTETRIS-4DGS本文のCC BY 4.0表示を確認したほかは未確認とし、
既存台帳の別versionに対するlicense確認を流用しない。
本Viewerでの性能、WebGPU実装可能性、SPL4互換性、SPH物理量保持は、いずれも未検証である。

| 候補 | 一次資料・確認範囲 | 公開実装について確認した範囲 | Viewerで参考にする責務・重要な限界 |
|---|---|---|---|
| ResStreamGS | [arXiv:2610.21121v1本文](https://arxiv.org/html/2610.21121v1)：GOP内のstable／sensitive分割、dual-stream配信・復元 | code availability／固定release未確認 | 区間内で再利用する属性と更新する属性の分離、GPU常駐・転送量。画像残差・最適化で分割する手法であり、静止粒子や永続不変属性の判定ではない |
| SteadySplats | [arXiv:2610.05576v2本文](https://arxiv.org/html/2610.05576v2)：stochastic renderingの低分散化、空間・時間再利用。別に[PlayCanvas Issue #9810](https://github.com/playcanvas/engine/issues/9810)の提案本文 | Issueは確認時Open。著者のlocal prototype報告であり、merge済み製品機能・固定実装の確認ではない。論文codeの公開版は未確認 | 独立した近似rendererの将来比較。現行sort／alpha合成の同値置換としない |
| KoalaGS | [arXiv:2610.28615v1本文](https://arxiv.org/html/2610.28615v1)、[公式project](https://rdbch.github.io/koalags/)：学習・再配置と重要度順prefixによる連続LOD | projectは確認時「Code (soon)」。code監査未実施 | LODの画質分布・切替連続性・投影面積。任意の既存modelを並べ替えるだけで同じ効果になるとは確認していない |
| SplatQ | [arXiv:2610.13503v1本文](https://arxiv.org/html/2610.13503v1)：地域保持quota、同時削除の相互作用を考慮するQUBO、patch画質分布 | code availability／固定release未確認 | 局所品質を守るoffline pruningと評価指標。fine-tuningを含む手法であり、Viewerの即時LOD制御ではない |
| TETRIS-4DGS | [arXiv:2610.14663v1本文](https://arxiv.org/html/2610.14663v1)：true 4D Gaussianの重要度別量子化・属性配置・PNG／HEVC符号化 | code availability／固定release未確認 | 4D属性の配信用派生payload候補。pruning／量子化を含み、SPL4のlossless圧縮・直接互換を意味しない |
| CCT-GS | [arXiv:2610.26920v1 abstract／metadata](https://arxiv.org/abs/2610.26920v1)：SHをcodec向けに変換するSH-ACとscene適応 | 本文詳細・code availability／固定release未確認 | 高次数SHの将来圧縮候補。学習された逆変換等を要し、通常の動画decoderだけで完結する方式とは扱わない |

#### Viewerへの応用候補と評価時の区別

以下の「評価したい点」はViewer側の設計判断であり、論文が本Viewerで実証した結果ではない。
既存Section 7、8、9.2、10の責務・評価軸を再利用し、現段階では数値閾値、codec、
buffer構成、分類方式、新APIを決めない。

- **ResStreamGS：不要更新の削減。** GOP内で再利用できる成分をGPUへ保持し、変化する成分だけを
  配信・更新する考え方をSection 7の候補に加える。論文の安定性は区間と属性に依存し、
  位置だけでは決まらない。既存のtrue 4D parameterを常駐させ時刻評価する経路と、
  frameごとの3D Gaussian更新経路を区別し、現在どこで転送が発生するかを測る必要がある。
  単純な静止／移動の二分法や、追加の画像学習をViewer要件にしない。
- **KoalaGS／SplatQ：局所品質と実際の描画負荷。** global PSNRだけでなく、
  patchごとの品質分布、局所coverage、細部の欠落、LOD切替の不連続を評価する着想を得る。
  KoalaGSでは低LODに大きなGaussianが残り、投影面積・重なりによって描画が重くなる場合も報告される。
  点数削減率を速度改善率へ置き換えず、同条件でSection 10.4のtile references、overdraw proxy、
  GPU時間・memoryと併せて評価する。SplatQの画像上の地域quotaを、物理領域の保存保証へ読み替えない。
- **TETRIS-4DGS：4D属性を保った配信圧縮の候補。** 時空間mean、4D rotation／scale、
  時空間SHを扱う点は参考になる。ただし既存SPL4や現在のactive SH0条件と同一形式ではない。
  attributeを配置したcodec frameとシミュレーション時刻を混同しない。PNGで保存する成分も、
  その前段の量子化を含めて元のfloat値への完全可逆性を評価する必要がある。
  render payloadをcanonical inputから派生させるSection 8の境界を維持する。
- **CCT-GS：高次数SHを使う場合の後段候補。** 現在のSH0比較に対する優先度は低い。
  将来評価では圧縮率だけでなく復元model・side information、decode／逆変換時間、
  CPU／GPU一時容量、random accessを含める。SH圧縮の有用性から高次数SHの導入を要求しない。
- **SteadySplats：描画方式を変える独立比較。** stochastic／履歴再利用を含むため、
  現行の決定論的CUDA参照と同じpixel値になることを前提にしない。近似方式として静止時のnoise、
  camera操作・時刻移動・scene切替後の収束と残像、history破棄、追加GPU memory、対応view／deviceを評価する。
  論文の低分散化、任意のtraining regularization、PlayCanvasのprototypeを区別する。
  Issueの限定scene／GPUの報告値をViewerの性能見込みにしない。Step122のsample／alpha／RGB／T契約や
  現行sortの修正へ混ぜず、必要性が確認された場合の別設計候補に留める。

#### SPH直接変換との接点

今回確認した範囲では、SPH kernelからGaussian covariance／weightを直接構成し、
元の物理場を再構成する問題をこれらの論文が解決したという根拠はない。
直接変換の定式化・保存量・近傍関係・物理的誤差評価は別projectの責務のままとする。
次は既存Section 8とdirect converterとのinterfaceへ持ち込める検討案であり、論文のSPH実証結果ではない。

- particle IDとtime identityを使い、位置・形状・表示色／alpha・物理属性の更新要否を属性ごとに区別する。
  低速度・不動の粒子でも物理量や表示属性は変化し得る。Gaussianとの対応と再利用条件は別途明示する。
- render payloadの画像誤差と、Scientific payloadの単位付き物理量誤差・粒子対応を別々に扱う。
  学習済みSH外観係数をpressure／density等の物理属性の代用にしない。
- 局所品質評価の着想を、関心領域での物理量誤差やcoverageの評価へ拡張できるか検討する。
  画像patchのPSNRやpruning成功だけで、物理量・粒子identity・保存量の保持を保証しない。

これらの候補登録はViewerのPhase 4開始、SPHをViewerの「Phase 5」とする新設、
direct converter側の設計変更、追加学習、外部rendererの採用を意味しない。
優先順位は既存Section 3.2の必要性判断と、Section 9.2／10に基づく将来の比較で決める。
Corrected採用範囲の完了、採用B／SPL4／CUDA Reference、Step21の入力受入れ・SH0接続、
Step122の限定比較・改良契約は変更しない。

<a id="literature-20261030-candidates"></a>

### 9.8 2026-10-30文献補足：Spark Radix・時間依存表現（未採用・未実測）

相談役取得日・CODEX保存資料確認日はいずれも2026-10-30 JST。
[一次資料台帳](../../reports/webgpu-4dgs-viewer/literature-20261030-docsync1/literature-20261030-docsync1-sources.json)に
URL・固定版・bytes identityを記録し、同directoryの`sources/`に本文を保全している。
SOG v3の訂正は[既存候補節](#sog-v3-candidate)へ統合した。以下はSection 9.2／10の比較材料であり、
既存候補の優先順位・Stageを変更しない。Section 3.2の必要性判断と別途認可を維持する。

| 候補 | 一次資料・固定版・確認範囲 | 公開実装／licenseと未確認事項 |
|---|---|---|
| Spark Radix | [公式repository](https://github.com/sparkjsdev/radix)の固定commitに対応するREADME、docs/sorting.md、docs/rendering.md、docs/bench.html、package.json。固定commitは上記台帳、packageは0.0.1。[公式入口](https://sparkjs.dev/radix/) | 公開reference implementation、root LICENSEはMIT。renderer source・依存全体のlicense監査・独立実行は未実施。サイトbuild日を初公開日としない |
| GaussianSpline | Tom Yachil / Andrei Sharf、[arXiv:2610.35224v1](https://arxiv.org/abs/2610.35224v1)、2026-10-26提出。abstract／metadataのみ | 本文詳細、code availability／固定release、code license、独立再現は未確認。保存metadataのlicenseリンクは一般案内であり、当該論文のlicense種別は未確認 |

**Spark Radixの用途と境界。** World LabsによるThree.js WebGPURendererベースの独立実装であり、
既存Spark系列の台帳とは区別する。公式文書が示すGPU compaction→stable radix sort→indirect draw、
WebGL2の非同期Worker sort、CPU/GPUで共通のTypedArray tableとCPU-only層の分離を、Section 4／7の
性能・ownership・fallback比較へ接続する。ただし公式sort keyは視点からの距離（視線方向のdepthではない）で、
遠→近の描画順、既定24-bit／16・32-bit選択である。Viewerの正の前方depth・full-precision／同値順・
近→遠C/T契約とは異なり、方向だけ移植せず距離定義・key精度・合成式を組として比較する。
Radix画像をStep122 CUDAの代替oracleにしない。

[著者benchmark](https://sparkjs.dev/radix/docs/bench.html)は2026-10-09、M3 Max／Chrome154、
Spark2.3.1／PlayCanvas2.23.1との比較で、公開日やSpark2.4.0の測定ではない。degree-0入力の当時の測定へ
現在のRay／SH等の機能を遡及して含めない。将来のViewer評価は同入力・camera・viewport・culling／blur・SH・
device／browserを揃え、sortだけでなくframe時間・一時容量・WebGL2の非同期遅延を扱う。
true 4D／SPL4の直接互換、科学属性保持、device portability、本Viewerの改善効果は未検証。
TSL共通化やWebGL2 fallbackの存在はbackend一致を保証せず、renderer置換・Three更新・LoD導入を認可しない。

**GaussianSplineの用途と境界。** abstractはGaussianごとの非一様な区分3次Bézier、制御点と時間分割の
共同学習、隣接区間のG1接続を提案し、非一様な動きの再構成品質改善を主張する。これは著者主張であり、
Viewerへの効果は未検証である。受取り側の候補評価としてSection 6／7へ、時刻区間検索、可変長制御点の
容量・配置、GPU常駐、評価cost、random seek、区間境界の連続性を接続する。既存の常駐true 4D時刻評価、
frame属性列、固定次数運動表現との比較は未検証案であり、現Viewerが毎frame全位置転送を要するとは仮定しない。
G1は接線方向の幾何学的接続であり、元SPH軌跡や時刻に対する速度・物理量の厳密保存を保証しない。
学習方式・converterの定式化、制御点生成、追加学習、SPL4再設計は対象外とする。

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

Issue #40による追加計測候補は、LOD探索/選択時間と、取消・無効化された処理の転送量である。
後者は無効化後にも到着・転送したbytesを区別して、Section 7の不要処理停止・結果破棄・再転送防止を
評価するための候補とする。既存stage計測を置き換えず、具体telemetry schemaや閾値は確定しない。
上流の高速化率は本Viewerの実測値・見込み値・受入閾値ではない。

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
