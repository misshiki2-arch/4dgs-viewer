# Phase 3 Current State

更新基準: 2026-09-01 / Phase 3 Step122 Investigation10 Documentation Sync時点
対象ブランチ: `phase3-webgpu-compute-prototype`
基準コミット: `c137ae4 Step121: complete 524288-record semantic parity`

## 1. この文書の役割

この文書は、4DGS Viewer の WebGPU 化について「現在どこまで完了し、何をまだ主張していないか」を示す現在地の正本である。

- 長期設計と各 Step の詳細は `docs/phase3_webgpu_backend_design.md` を参照する。
- WebGL2 で到達した自然な境界は `docs/phase2_webgl2_closure_summary.md` を参照する。
- 初期の責務分担案は `docs/phase3_backend_responsibility_plan.md` に履歴として残す。
- 上記文書の古い `Current` / `Next` / `current goal` 表記より、この文書の現在地を優先する。
- 次の Step の目的は、ユーザーが設計方針を確認してから別途確定する。この文書だけを根拠に実装へ進んではならない。

## 2. プロジェクト全体の大目的

4DGS の状態評価、可視判定、投影、tile reference、depth ordering、compositing、canvas presentation を WebGPU production backend が所有し、camera と time の対話操作を含めて CUDA 参照と意味的に整合する 4DGS Viewer を成立させる。

単に WebGPU canvas が nonblank になること、capture が成功すること、または一つの fixed frame が表示されることだけを最終目的としない。

この大目的はViewer project固有である。研究program全体の目的階層とprojectの役割は
[`50_4DGS_RESEARCH_PROGRAM_GOALS_JA.md`](../../4dgs-development-governance/50_4DGS_RESEARCH_PROGRAM_GOALS_JA.md)
を正本とする。Viewerは直接変換結果の利用・検証基盤であり、研究programの中心成果またはdirect converterのownerではない。

direct converterとはcanonical format、manifest、provenance、粒子identity、時刻、物理属性等の
明示的interfaceで接続する。ViewerのPhase／Stepをconverterまたは研究programへ流用しない。
LOD、streaming、compressionは現在の必須実装ではなく、real-time利用が実測上成立しない場合に限り、
別途認可する独立責務として再評価する。

## 3. 維持する責務分担

- WebGPU は将来の標準 production backend を所有する。
- WebGL2 は fallback、回帰確認、比較 oracle として独立して維持する。
- 同一 production frame 内で WebGPU と WebGL2 の結果を混成しない。
- Three.js / viewer shell は UI、camera 入力、runtime wiring を担当し、renderer semantics の owner にはしない。
- fixed CUDA reference camera と通常の interactive camera は別モードとして扱う。
- production runtime、capture tool、Summary tool、描画数式、camera / projection、性能改善は別責務として変更する。
- diagnostic observer は production ownership を変更しない。
- browser でユーザーが確認した事実を Summary や artifact より下位に扱わない。

## 4. Phase 2 の到達点と停止線

Phase 2 では WebGL2 で自然に成立する範囲を確定した。

主な到達点:

- GPU screen-coarse candidate generation
- validated-only promotion と CPU fallback
- raw 4DGS attribute texture と shader `texelFetch`
- Transform Feedback による fixed record 生成
- radius、conic、alpha、tile range を含む fixed-record 比較
- packed-like dry-run と差異分類
- display connection readiness の観測
- CPU post-candidate workload の計測

WebGL2 へ無理に持ち込まないと判断した範囲:

- variable-length compaction と prefix sum
- scalable depth sort
- GPU tile list / scatter
- per-tile ordering
- production tile compositor
- sorted production display connection

これらが compute-oriented であるため、Phase 3 は WebGPU を中心に進めている。WebGL2 は削除せず fallback / oracle として維持する。

## 5. Phase 3 の進行概要

Phase 3 は次の責務を段階的に構築してきた。

1. common contract、WGSL、GPU buffer、compaction、ordering、tile list の基盤
2. frame lifecycle、runner、executor、uniform、bind group、output、handoff、`currentTexture` ownership
3. GPU-owned visible generation、4D state、attribute、footprint、tile input、tile list、depth ordering、tile compositor
4. time、camera、viewport、dirty scheduler、workload budget、resource lifecycle、work reduction
5. fixed CUDA reference、camera evidence、固定条件比較、orientation、projection、covariance / Jacobian / conic parity
6. capture lifecycle、diagnostic nonmutation、final-writer ownership、artifact / Summary の信頼性
7. viewer state から native WebGPU production frame を生成し presentation まで接続する data path

Step 番号は実装責務を小さく分けるための進行単位であり、番号の大きさ自体を完成度の証拠にしない。

## 6. Step116 から Step118 の位置付け

### Step116

generic capture lifecycle verification を整備し、fresh production frame を対象に diagnostic JSON / PNG を生成して判定する経路を確立した。

### Step117

diagnostic ownership と capture artifact を再設計し、production final-writer、generation、request identity、last-valid output cache、presentation freshness を diagnostic observer から分離した。

### Step118

viewer state から次の native WebGPU production data path を接続した。

```text
viewer state
-> 4D state evaluation
-> attributes
-> projected footprint / tile payload
-> production tile input
-> GPU-owned tile references
-> bounded tile sort / compositor
-> output texture
-> currentTexture presentation
```

Step118 の browser 確認と machine-readable Summary では、同一 request / generation について production frame completion、compositor output、canvas write、presentation、fresh nonblank PNG、artifact identity が成立した。Step117 の ownership / presentation 契約も維持された。

## 7. Acceptance level

`ready` の意味を混同しないため、到達状態を次の段階に分ける。

| Level | 意味 |
|---|---|
| 1 | mechanism ready: 個別機構が契約どおり動く |
| 2 | ownership / lifecycle ready: owner、generation、request、diagnostic 境界が成立する |
| 3 | native production data path ready: production 入力から final canvas まで native WebGPU 経路が通る |
| 4 | fixed-condition visual / semantic parity ready: 固定条件で CUDA 参照と意味的に整合する |
| 5 | interactive production acceptance ready: camera / time 操作を含む通常 runtime が成立する |
| 6 | performance / scalability ready: scene 規模と対話性能の目標を満たす |

Step118 完了時点の Level 3 は引き続き成立する。Step121は固定条件における
record-local semantic parityを閉じ、Step122 Impl1 Fix2はproduction tile-input alphaを実browserで
0 / 0 mismatchへ到達させた。Investigation7は次のRGB mismatchをexpected oracleとproductionの
独立した二問題へ分類し、Impl1 Fix3はexpected-oracle C2符号だけを修正して独立browser gateを
acceptedにした。Investigation8はproduction spatial SHに先立ってdiagnostic RGB / alpha validityを
分離すべき独立consumer不具合を確定し、Impl1 Fix4はその分離を実装して自動検証を完了した。
Step122 Impl3はproduction evaluatorへdegree-2 spatial SHを実装し、固定524,288 recordsの15-stage
browser semantic gateをalpha / RGBとも0 mismatchでacceptedにした。Validation2はpost-Impl3 fresh
PNGを当時のfixed CUDA Referenceと比較し、MAE / RMSE改善と未分類の残差を確認した。Investigation10は、
そのCUDA Referenceでfull projectionとfootprint rasterizerのintrinsicsが分裂していたことを確定した。
したがって13 / 15-stage gateはWebGPU actualとcurrent expected oracleの内部整合として維持する一方、
CUDA-direct screen / raster parityは再検証待ちである。fixed PNG parity、downstream sort / compositor /
final RGB、full-scene gateが残るため、Acceptance Level 4の完了とは扱わない。Investigation10の技術詳細は
Section 22、現在のfreeze境界と依存順はSection 23を正本とする。

## 8. Step118 が証明したこと

- native WebGPU production data path が viewer state から compositor / presentation まで接続された。
- bounded execution と overflow fail-closed contract が成立した。
- production tile-reference count の machine-readable 整合が確認された。
- production final-writer と diagnostic nonmutation の責務が維持された。
- request identity、production generation、presented generation が一致した。
- final canvas write が attempted / submitted / completed となった。
- capture PNG は fresh、nonblank で、browser に継続表示された内容と一致した。
- Step118 Summary の Step118 decision は `ready` である。

## 9. Step118 が証明していないこと

- CUDA reference PNG との最終的な画素または構造的一致
- Gaussian の位置、形状、色、opacity の最終正当性
- SH / color evaluation の最終 parity
- depth ordering と alpha blending の最終 parity
- scene 全体または全 visible population の完全処理
- time を変化させたときの 4D semantics
- interactive camera / time での通常 runtime acceptance
- production 性能と scalability
- LOD、streaming、early termination などの将来機能

Step118 browser 確認で得られた赤紫色の全面表示は、stable nonblank production presentation の証拠である。一方で、CUDA 参照と意味的に一致する Gaussian 表示の証拠ではない。この観察を Step118 の失敗とも visual parity の成功とも扱わない。

## 10. 現在残る作業分類

Step121は固定resident range `[524288, 1048576)`の全524,288 recordsについて、WebGPU actualと
current expected oracleの定義済み13 stageを比較するinfrastructureとinternal semantic matchを完了した。
Step122はfresh PNG比較を実行し、
comparisonを15 stageへ拡張してproduction tile-input alphaを修正した。Impl1 Fix2のbrowser artifactは
21,818 eligible recordsのalphaを0 records / 0 components mismatchとして受け入れ、RGBを最初の
semantic / downstream mismatchへ進めた。Investigation7はRGBについて、expected oracleのC2符号と
当時未実装だったproduction spatial SHを独立した問題として確定し、Impl1 Fix3は前者だけを修正してbrowser
acceptedとなった。詳細とartifactはSection 21を正本とする。

Investigation8は、canonical lower-only spatial SHが正当なRGB > 1を生成し得るのに、当時のconsumerが
RGBとalphaを共有`colorAlphaValid`で判定し、RGB全成分へ`0 <= rgb <= 1`を要求していたことを確認した。
Step122 Impl1 Fix4はImpl3に先立ってvalidityを分離し、実装・自動検証を完了した。Impl3はその後、
production degree-2 spatial SHと独立browser semantic gateを完了した。次はValidation2時点の履歴的な
依存順であり、現在のfreeze境界と依存順はSection 23を正本とする。

1. **完了済み**: Step122 Impl1 Fix3でcomparison expected oracleのdegree-2 C2符号だけをCUDA正本へ同期する。
2. **完了済み**: production sourceを変更せず、同じ固定524,288 recordsの15-stage browser comparisonでexpected-only変更を受け入れる。
3. **完了済み**: Step122 Investigation8でdiagnostic RGB / alpha validityの独立不具合を分類する。
4. **完了済み**: Step122 Impl1 Fix4でvalidityを分離し、52-case focused smokeと関連回帰を完了する。Fix4は独立browser gateを持たず、Impl3 gateで非干渉を確認する。
5. **完了済み**: Step122 Impl3でproduction evaluatorへfixed baselineのdegree-2 spatial SHとversioned candidate input layoutを実装する。
6. **完了済み**: Impl3の15-stage browser comparisonでproduction RGB / alpha semantic residual 0、完全coverage、diagnostic device分離を確認する。
7. **完了済み**: Step122 Validation2でpost-Impl3 fresh fixed-condition PNGを同じCUDA Referenceと比較し、改善した誤差と残る未分類差を記録する。
8. **当時の次候補（Investigation10により保留）**: productionとCUDAのtile-reference population / countを最初に比較し、その後はdepth key / sort、per-tile ordered reference list、compositor accumulation、final RGB / PNG encodingを最初の不一致stageが示す順に分類する。
9. fixed-range downstream parityの後、3,231,588 recordsのfull-scene correctness gateを設ける。device limit内のfull residency、または全source populationをsilent omissionなく扱うcorrectness-preserving chunk / streamingで、同条件のCUDA full-scene Referenceと比較する。
10. interactive camera / time、resource lifecycle、performance、scalabilityを後続の独立責務として確認する。

Fix3 browser acceptanceの目的はproduction改善またはvisual parityではなく、corrected expected oracleが
固定524,288件でcorrected baselineを生成することの独立確認である。accepted artifactではalpha 0 / 0、
upstream 13-stage semantic residual 0、RGB 21,030 records / 47,529 components、最大
`0.9841542530436316`、first mismatch srcIndex 659525 / Rで、first semantic / downstream mismatchは
`productionTileInputRgb`である。production sourceは不変で、ユーザーのViewer目視も正常だった。
Fix3ではPNGを保存していないため、PNG identityまたはvisual parityを主張しない。

上記は候補と依存関係の記録であり、新しいImpl / Fixまたはproduction変更をこの文書だけで認可しない。

## 11. 証拠と判定

Step の完了判定では、次を区別する。

- source / design contract
- static smoke / build
- production runtime evidence
- capture command contract
- Summary decision
- PNG と JSON artifact
- ユーザーの実 browser 観察

一つの `ready`、nonblank、error absence だけで上位の acceptance level を主張しない。Summary と browser 観察が矛盾する場合は、どちらかを捨てずに原因を調査する。

### CUDA Reference／fixed-condition visual comparisonのPython環境

CUDA Referenceの生成・再生成と再検証、CUDA Referenceに関係するfocused test、fixed-condition
PNG定量比較、pixel evidence生成では、既知正常のconda環境`4dgs310`を使用する。ユーザーが行う
関連testも同じ環境を使用し、再現可能な手順ではshellのactivation状態へ暗黙に依存せず、次の
absolute executableを正本とする。

`/home/demo/miniconda3/envs/4dgs310/bin/python`

現在read-onlyで確認したbaselineはPython 3.10.14、NumPy 2.2.6、Pillow 9.4.0である。これらは
現在のenvironment evidenceであり、永久固定するversion requirementではない。実際にCUDA
Referenceを生成したPython、PyTorch、CUDA、rasterizer等のversionとenvironment provenanceは、
各CUDA render-state manifestを実行ごとの正本とする。現行のStep119 Impl7 manifestは
`condaDefaultEnv: 4dgs310`と上記executableを記録している。

通常shellの`/usr/bin/python3`にNumPyがあるかどうかを、このworkflow全体の利用可否判定に
使用しない。version driftだけを理由に環境を変更せず、必要なimportまたは既存toolが失敗した場合は
そのerrorを報告して停止する。package install、new venv、interpreterまたは既存conda環境の変更には
別途認可が必要である。別用途のtoolが明示的に別environmentを要求する場合は、そのtool固有contractを
優先し、`4dgs310`を黙って置き換えない。

Step122 Validation1とImpl2のfixed-condition PNG比較でも、この既知環境を再利用した。この
environment確認や比較実行だけで、Step122完了またはCUDA/WebGPU visual parityを意味しない。

## 12. 設計文書の管理

- 現在地と次 Step の設計は、設計管理担当が会話履歴、Git、設計書、artifact、browser 観察を照合して更新する。
- CODEX は source 調査と実装結果から設計上の影響を報告できるが、明示的な文書更新命令がない限り現在地や次 Step の目的を独自に書き換えない。
- runtime 修正と設計文書の大規模整理を同じ命令へ混ぜない。
- 過去の設計文書は削除せず、historical baseline として維持する。
- 次 Step の設計は、元の不具合、大目的、成功条件、非対象、変更責務、維持契約、停止条件を固定してから追加する。
- CODEX 命令文の作成と運用は `/mnt/c/Users/isshi/CodexRules/CODEX_COMMAND_CREATION_AND_OPERATION_FIXED_RULES_JA.txt` に従う。


## 13. Step119 の目的

Step119は、固定camera・固定time・同一source populationでCUDA Referenceとproduction WebGPUを比較できる最初のsemantic controlを成立させるStepである。PNG全体のvisual parityを先に主張せず、population、camera、time、temporal visibility、production completion、orientationを一致させてから、最初に異なる下流責務境界を確定する。

Step119では二つのcontiguous source rangeを使う。

- `[0, 524288)`はzero-visible controlであり、CUDA zero-visibleとWebGPU ready-zero / production blackをruntime failureから区別する。
- `[524288, 1048576)`はvisible controlであり、同じ524,288 recordsをCUDA Referenceとproduction WebGPUのresident worksetへ適用して、下流のfixed-condition visual / semantic comparisonを行う。

いずれもscene全体residencyを意味しない。元sceneは3,231,588 recordsである。このfixed-range比較は同じresident population内のsemantic parityには使用できるが、非resident recordsがcoverage、depth ordering、alpha accumulationへ与える影響を含まないため、full-scene visual parityを証明しない。fixed-range semantic parityの後には、full residencyまたはcorrectness-preservingなchunk / streamingにより全source populationをsilent omissionなく処理し、同条件のCUDA full-scene Referenceと比較する必須gateを置く。このgateを通過するまで、最終viewerのvisual parityまたはproduction acceptanceを宣言しない。LODを使う場合は完全population処理と自動的に同一視せず、近似内容とacceptance contractを別途定義する。interactive camera / time、performance / scalabilityも後続責務である。

## 14. Step119 の確認済み到達点

### Fix1 / Impl2 / Impl3: zero-visible populationとtemporal eligibility

checkpointとSPL4-v2のsource prefix `[0, 524288)`、524,288 records、136,314,880 bytesがbyte-exactに一致することをacceptedとした。checkpoint range SHA-256とasset range SHA-256はともに`67b72bcf2a307ea21ea4185ce95d54d426ff130e47726e19f147ff4be99b5633`であり、artifactは`/home/demo/work/json/phase3_step119_fix1_population_provenance.json`である。

accepted provenanceから同じsource prefixだけを既存CUDA production rasterizerへ入力した。camera `000151_v13`、time `23.2`でCUDAはzero-visibleとなる。Impl3はCUDA temporal marginal gateに対応するvisibility eligibilityをproduction WebGPU evaluatorへ追加し、ineligible rowをstate / attribute / footprintの各production payloadでnon-contributingにした。

### Impl4 / Impl5 / Impl6 / Validation1: zero-visible control accepted

Impl4はcanonical zero-work execution completionを実装し、positiveなresident inputからrequired / written / scattered / sorted / composited referenceがすべて0となる正常完了をfailureや未dispatchから分離した。Impl5はsuccessful production black / cached production blackをfallback / clear / failureから分離し、blackを同一identityのlast-valid production outputとして扱えるようにした。Impl6は明示的opt-in時だけzero-visible production-blackを受け入れるcapture policyを追加した。

Validation1ではbrowser観察とmachine-readable artifactが一致し、同一population / camera / timeのWebGPU ready-zero、production black、fresh black PNGをruntime failure、fallback、clear、stale outputから区別できた。zero-visible controlとImpl3のformal acceptanceは完了している。

### Impl7 / Impl8: CUDA-visible population control

Impl7はgeneric contiguous source range `[524288, 1048576)`、524,288 recordsをCUDA Reference generatorへ適用した。固定条件はcamera `000151_v13`、time `23.2`、1280 x 720である。canonical CUDA output directoryは`/home/demo/work/outputs/sph_scene_4dgs/cuda_reference_step119_impl7_population_0524288_1048576/iter_012000`であり、renderはnonblankである。

Impl8は同じrangeをproduction WebGPUのresident worksetへ適用した。scene count 3,231,588に対してresident count、state count、tile-input count、workset countはすべて524,288で一致し、resource lineageはproduction compositorまで維持される。これは一つの明示的resident rangeであり、full residency、streaming、LODの実装ではない。ユーザーのbrowser観察ではnonblank Gaussian表示が安定し、保存PNGと表示が一致した。

### Investigation6 Fix1 / Fix2: orientation contract accepted

Fix1のsame-index diagnosticはImpl7 CUDA direct evidenceと同じ8 recordsを比較し、temporal motion delta errorは0、camera-space最大誤差は`7.629e-6`、raw screen center最大誤差は`6.104e-5 px`であった。raw production outputとCUDA renderはいずれもtop-left / y-downで一致していたが、presentationとPNGだけがそれぞれ独立にvertical flipしていた。

Fix2はcommon presentation / capture orientation policyだけをtop-left / y-down、flipなしへ変更した。camera、projection、screen center、tile input、sort、compositor semanticsは変更していない。fixed条件のproduction / presented / captured generationはすべて3、captureはfresh、saved PNG SHA-256はBlobと一致し、RGB nonzero pixel countはFix1 / Fix2ともに174,968である。height 720に対しnonblack bounding boxはFix1 `[167, 53, 1021, 571]`からFix2 `[167, 148, 1021, 666]`となり、`y' = 719 - y`のexact vertical flip関係を満たす。file checkはrequired artifact present / JSON validで、WebGPU validation、queue submit、device lost errorはない。ユーザー確認でもviewer / PNGはCUDA / GTと同じ上下方向となり、Fix2はacceptedである。

### Validation2: fixed-visible RGB comparison accepted

Validation2はImpl7 CUDA renderとInvestigation6 Fix2 production WebGPU PNGを、source range `[524288, 1048576)`、524,288 records、camera `000151_v13`、frame `151`、time `23.2`、1280 x 720、top-left / y-downの固定条件でRGB 3 channelsとして比較した。input identityとcapture freshnessは成立し、runtime validation、Invalid CommandBuffer、queue submit、device lostの各failureはない。比較作業とartifact保存はacceptedだが、visual parityの成功ではない。

正式metricsはMAE `8.168661747685185`、RMSE `32.02683437269976`、max absolute error `255`、different pixel count `175123`、different pixel ratio `0.19002061631944445`である。normal orientationのRMSE `32.02683437269976`はvertical flipの`45.58795176589415`より低く、lower-error orientationはnormalである。CUDAのRGB nonblack count / bboxは`131093` / `[214, 157, 1031, 613]`、WebGPUは`174968` / `[167, 148, 1021, 666]`である。Validation2 artifactのcanonical classificationは`comparison-ready-difference-unclassified`であり、Y反転は残る差の原因ではない。artifact単独では未分類だった最初のsemantic stageは、後続のInvestigation7で分類した。

Canonical artifacts:

- `/home/demo/work/json/phase3_step119_validation2_000151_v13_fixed_visible_comparison.json`
- `/home/demo/work/json/phase3_step119_validation2_000151_v13_fixed_visible_absdiff.png`
- `/home/demo/work/json/phase3_step119_validation2_000151_v13_cuda_pixel_evidence.json`
- `/home/demo/work/json/phase3_step119_validation2_000151_v13_webgpu_pixel_evidence.json`

### Investigation7: first semantic mismatch分類完了

Investigation7は同じ8 srcIndexについて、既に一致していたraw screen centerより上流から順にsemantic stageを確認した。population / source identity、scale XYZ、scaleT、normalized left/right quaternion、Gaussian time、evaluated time、4D rotationと`Sigma_tt` / `Sigma_12` temporal coupling、raw screen centerは許容範囲内で一致する。

Investigation7で最初に確認された不一致は、4D Gaussianからconditional 3D covarianceをproduction footprintへ渡す境界だった。CUDA production rasterizerは左右quaternionとXYZT scaleから4D covarianceを生成し、`Sigma11 - Sigma12 Sigma12^T / Sigma_tt`を使用する。当時のWebGPUはtemporal meanでは左右quaternionとscaleTを使用する一方、production footprint covarianceでは左quaternionとscaleXYZだけから通常の3D covarianceを生成していた。この不一致は後続のStep120で解消済みである。

captured inputからCUDA式を再構成したconditional 3D covarianceは8/8でCUDA direct値へ`1e-5` tolerance内で一致し、当時のWebGPU qL-only式は8/8で同toleranceを超えた。CUDA covarianceはproduction rasterizer debug rowのdirect evidenceである一方、当時のWebGPU conditional world covariance自体はartifactへactual GPU intermediateとして直接保存されておらず、captured production inputと当時のWGSL式から再構成した。このevidence provenanceをdirect GPU evidenceとして扱わない。

Investigation7のfirst-mismatch停止時点では、camera-space covariance、projection Jacobian、screen covariance、conic、radius / footprint、opacity / SH color、tile coverage、depth sort、compositor accumulationは未判定だった。後続のStep120とStep121が確定した範囲は各完了節を正本とし、それ以外を一致済みまたは問題なしとは主張しない。

### `canonicalExecution.ready` field scope分類完了

Fix2 captureのpolicyは`production-nonblank`で、actual work classificationは`nonzero-reference`である。`captureExpectationContract.canonicalExecution.ready`はgeneric execution completion全体ではなく、zero-reference opt-in経路で使う`genericCanonicalZeroExecutionReady`を格納する。そのためFix2で`false`なのは期待どおりであり、runtime failure、stale execution evidence、またはFix2 orientation regressionではない。

同じcapture expectationは`readyBeforePng: true`、`ready: true`、`blockedReason: null`であり、runtime terminal contractの`executionCompletionReady: true`と整合する。既知のevidence inconsistencyとされていた項目は、field-scopeの分類により解消した。

## 15. Step119 Investigation4 で確定し、解消した設計不備

Investigation4時点では、execution completion、nonzero work、output write、pixel color、presentation path、acceptance policyが一つの`ready`または`success`へ重なっていた。そのため正常なzero-reference executionとproduction blackをfailureから区別できなかった。

Impl4からImpl6はこの不備を、canonical execution completion、work classification、output completion、presentation classification、pixel classification、acceptance policyへ分離した。Validation1で実artifactまで受入済みであり、この履歴課題を未実装の次作業として扱わない。

## 16. Canonical production outcome 設計

production outcomeは次の直交する事実として扱う。

1. **Execution completion**: pending / completed / failed。plan status、identity、static shape、compact offset readiness、stage count一致、overflowで判定し、reference数や画素色には依存しない。
2. **Work classification**: zero-reference / nonzero-reference。source population自体が0の場合は正常zero-workへ昇格させない。
3. **Output completion**: output passとtexture writeが完了したか。zero-referenceでも意図したblack outputを書けばcompletedである。
4. **Presentation classification**: production / cached production / fallback / clear / failure。event kindはwriter、path、causeから決め、画素色から決めない。
5. **Pixel classification**: black / nonblank / unknown。正常なproduction presentationとblackは両立する。
6. **Acceptance policy**: 既定policyはnonblank productionを期待し、明示的opt-in policyだけがzero-reference production blackを期待する。runtimeはfactsを公開し、capture / Summary consumerがpolicyと照合する。

last-valid outputの`valid`はnonblankではなく正常に完了したproduction outputを意味する。正常なproduction blackは同一identityのlast-valid sourceになり得るが、fallback、clear、failed / stale outputはcacheを置換しない。final-canvas boundaryはwrite completion、source identity、path、pixel result、persistence、overwrite、quiescenceをpolicy-neutralなfactsとして保持する。

## 17. Step119 の分類gate完了とStep120

### Validation2: read-only fixed-visible image comparison

このread-only比較はaccepted済みである。canonical targetはImpl7 CUDA `000151_v13_render.png`、actualはInvestigation6 Fix2 WebGPU `phase3_step119_investigation6_fix2_000151_v13_canvas.png`であり、comparison artifactと両入力のpixel evidence、absdiff PNGを保存した。固定条件とruntime / capture healthは比較可能だったが、約19.002%のRGB pixel差が残るためvisual parityではない。

### Investigation7: first downstream mismatch classification

分類は完了した。first semantic mismatchはconditional 4D-to-3D covariance production-footprint境界である。source/index、temporal inputs、4D temporal coupling、raw screen centerはその前段で一致し、camera covariance以降のstageは未判定である。

### Capture expectation field scope

`canonicalExecution.ready: false`はzero-reference専用predicateをnonzero-reference captureへ表示した結果であり、runtime completionとの不整合ではない。capture expectation全体とruntime terminal completionはともにreadyであるため、tool、schema、artifactの修正をStep119の残存gateにはしない。

### Step120（Investigation7時点のhistorical plan）

以下はStep119 Investigation7完了時点の計画記録である。Step120は後続のImpl1、Impl2、Impl2 Fix1-Fix4によって完了しており、現在の実装許可として読まない。完了内容と境界はSection 19を正本とする。

当時のproduction実装計画は、既存のCUDA-aligned 4D rotation / covariance vocabularyを利用してconditional 3D covarianceをproduction footprintへ接続する一責務に限定していた。camera transform、projection Jacobian、Y orientation、conic、radius、tile coverage、sort、compositor、opacity、SH/colorを同時に変更せず、修正後はcamera-space covarianceから比較を再開して次の最初のmismatchで停止する方針だった。Step120だけでvisual parity成功を予定事項として断定しない境界も維持された。

## 18. Step119 の到達点と非対象

Step119はfixed-population semantic controlとfirst-mismatch classificationの責務として閉じる。これはfixed-range visual parityまたはAcceptance Level 4の達成を意味しない。

完了済み:

- fixed visible comparison artifactが保存されている。
- CUDAとWebGPUでpopulation、camera、time、resolution、orientationが一致している。
- runtime / capture failureとrendering differenceが分離されている。
- screen centerより上流の一致と、最初のconditional covariance mismatchが分類されている。
- `canonicalExecution.ready`がzero-reference専用fieldであり、nonzero-reference runtime completionと矛盾しないことが分類されている。

Step119完了時点で未達だった項目（historical。先頭2項目の現在地はSection 19を参照）:

- conditional 4D-to-3D covarianceのproduction実装
- conditional covarianceより後段のsemantic comparison
- fixed-range visual parityとAcceptance Level 4
- 3,231,588 recordsのfull-scene correctnessとmatched full-scene CUDA comparison
- final production acceptance

Step119ではcamera、projection、temporal、tile、sort、compositorの変更を行わなかった。scene全体residency、streaming / LOD、interactive camera / time、performance、retry / RAF / heartbeat、大量trace、Step / camera / frame / time依存のproduction分岐も非対象だった。最初のmismatchに対する修正は後続Step120で一責務として完了した。

この非対象指定はStep119 Investigation7へfull-scene実装を混ぜないためのものであり、最終受入れから除外する意味ではない。52万件のfixed-range semantic parityを閉じた後、3,231,588 recordsのfull-scene correctnessを独立した必須gateとして設計・検証する。

## 19. Step120 Conditional 4D-to-3D Covariance Production Footprint

### 完了した実装責務

Step120はStep119 Investigation7で確定した最初のsemantic mismatchだけを対象にし、Impl1からImpl2 Fix4までを完了した。

- **Impl1 — production semantics**: production WebGPU footprint covarianceをqL-only / XYZの通常3D covarianceから、左右のnormalized quaternionとXYZT scaleで4D covarianceを構築して`Sigma11 - Sigma12 Sigma12^T / Sigma_tt`を評価するCUDA-aligned conditional covarianceへ変更した。temporal meanとfootprintは同じ4D covariance vocabularyを参照し、production evidenceは`sourceCode: 113`を使用する。camera transform、projection Jacobian、conic、radiusのdownstream式は変更していない。
- **Impl2 — diagnostic expected consumer**: expected側もraw asset/build configurationから同じqL/qR/XYZT conditional semanticsを独立に導出するよう同期した。actual GPU intermediateをexpected計算へ入力しておらず、diagnosticはproduction controlにならない。
- **Impl2 Fix1 — canonical serialization**: boundedなStep113 semantic evidenceをDesign C canonical diagnosticの`$.stageSummaries.tileCompositor.step113SemanticEvidence`へ直列化した。representative上限は16であり、production runtime、readback、artifact schemaを変更していない。
- **Impl2 Fix2 — representative identity**: canonical source indices `658947`, `771007`, `788034`, `826401`, `835183`, `852955`, `863505`, `906711`をevaluator rows `0..7`へauthoritativeに対応付けた。固定fraction由来のtemporal-ineligible選択を除去し、JS/WGSL row identityを一致させた。
- **Impl2 Fix3 — fail-closed classification**: missing/invalid readbackを有効値へ昇格させず、そのrowのerrorsをnullにする。required evidenceが揃うまで4分類を`missing`に保ち、error aggregateは有効値だけから計算する。canonical browser captureでは8件すべてvalidであり、missing/invalid branchはfocused smokeで確認済みである。
- **Impl2 Fix4 — fixed-record validity**: fixed-record actual producerはstate unavailable / temporal-invalid rowをraw source値から再びvalidへ昇格させない。raw reserved/direct evidenceは維持する一方、compared validにはstate availabilityを必須とした。production evaluator、resident workset、tile/sort/compositor、presentation、captureは変更していない。

### Fix4 browser acceptance

Fix4 browser captureはsource range `[524288, 1048576)`、524,288 resident records、camera `000151_v13`、frame 151、time 23.2、1280 x 720、top-left/y-downの固定条件で、Fix3と同じviewer表示・runtime baselineを維持した。required artifactはすべて存在してschema/loadが正常、capture command contractは`ready`、canonical diagnostic statusは`ok`、exception / fatal error、WebGPU validation error、invalid command、queue failure、device lossはなかった。PNGはfreshかつnonblankで、requested state、presented frame、captured imageのidentityおよびBlob/saved-file identityが一致する。SHA-256はFix3と同一の`f843845d62e2377352cd998b3a09a5a15da4afdc60c23498c51795f2e594fec0`である。

canonical Step113 semantic evidenceはrepresentative 8件、completed 8件、missing 0件、invalid 0件で、rows `0..7`と上記8 srcIndexが一致する。全件が`sourceCode: 113`、`firstMismatchStage: none`であり、conditional/rotation classificationは`cuda-conditional-4d-to-3d-covariance-matched`、Jacobianは`cuda-aligned-camera-jacobian`、conic/radiusは`cuda-aligned-partial`である。`actualEvidenceSameProductionDispatch`はtrue、`productionCalculationDependsOnDiagnosticReadback`はfalseである。

8件について確認されたstage別最大絶対誤差は次のとおりである。

| Stage | Maximum absolute error |
|---|---:|
| conditional world covariance | `7.315222377846098e-8` |
| camera covariance | `5.944491415776909e-8` |
| projection Jacobian | `2.4726137084485345e-6` |
| screen covariance | `3.6210082640764085e-5` |
| conic | `8.152026688135194e-7` |
| radius | `0` |

したがって、このcanonical 8件に限りconditional world covarianceからcamera covariance、Jacobian、screen covariance、conic、radiusまでをacceptedとする。opacity/SH、tile coverage、sort、compositor accumulationへこの判定を拡張しない。

### Generic fixed-record comparisonのscope

Fix4のgeneric fixed-record artifactはcandidate / computed / comparedが各65,536件、actual valid 8件、expected valid 8件、`anyMismatch: false`、`fieldMismatchCount: 0`、`mismatchClassification: none`、`firstMismatches: []`である。最大絶対誤差は0ではなく`9.1552734375e-5`で、tolerance `1e-3`内にある。Fix4前の256,736 field mismatchは、64,184 invalid rowsを4 field（valid / px / py / depth）で誤って比較した結果であり、Fix4で解消した。

この65,536件populationはresident range `[524288, 1048576)`の先頭65,536件ではない。canonical 8 srcIndexを先頭へ追加した後、CPU fallbackのstride-1 indicesをdeduplicateして埋めるため、rows `0..7`はcanonical 8件、row 8はsrcIndex 0、row 9はsrcIndex 1、最終row 65,535はsrcIndex 65,527である。したがって「65,536件中valid 8件」を「resident 524,288件中visible 8件」または「残りのproduction resident recordsは正しくculled」と読み替えてはならない。この比較が証明するのはdiagnostic expected/actual validityとfixed field比較の整合だけである。

### Summary boundary

Design Cの`diagnosticExecutionDecision`は`ready`、`comparisonDecision`は`match`、`captureArtifactDecision`と`detailDecision`とoverall `decision`は`ready`、`blockedReasons`は空である。一方、legacy `step113CovarianceJacobianConicParity`はhistoricalなStep113 phase/rootを要求するため`blocked`のままである。これはStep120 runtimeまたはcanonical Design C artifactのfailureではなく、Summary consumerがStep120の責務境界へ完全には同期していないという別責務である。このDocumentation1ではSummary toolを変更しない。

### 完了境界と非主張事項

Step120が完了したのは、conditional covarianceのproduction実装、独立expected consumer、bounded canonical evidence、canonical 8件のrepresentative identity/fail-closed classification、およびgeneric fixed-record validityである。Fix4 browser acceptanceによりViewer、production presentation、JSON/PNG captureへの回帰がないことも確認した。

次は未達であり、Step120完了から推論しない。

- 全524,288 resident recordsの全stage comparison
- 全524,288 recordsのconditional intermediate comparison
- fixed-range visual parityとAcceptance Level 4
- opacity/SH、tile coverage、sort、compositor accumulationのsemantic parity
- 全3,231,588 source recordsのfull-scene correctnessとmatched full-scene CUDA comparison
- interactive camera/time acceptance
- performance、scalability、LOD、streaming
- final production acceptance

このStep120時点の将来候補は、後続Step121でbounded eight-chunk comparisonとして
実装・acceptance済みである。現在のStep121完了境界と未達gateはSection 20を正本とする。

## 20. Step121 Population-Aligned Record-Local Semantic Parity

### 実装構成とbounded ownership

Step121は、Step120時点では未確認だったproduction resident interval全体を、同一population・
同一順序・同一固定条件で比較するbounded diagnostic経路として実装した。

- single-chunk producerは最大65,536 recordsを処理し、packed evidence length、safe-integer row identity、stage validity確定後のaggregate commit、missing / invalidのfail-closed処理、CPU expectedとGPU actualの独立性を検証する。
- orchestratorはresident range `[524288, 1048576)`を8 x 65,536 contiguous chunksへ固定し、同じcaller-owned diagnostic deviceでsequential-awaitする。mismatchでは全8 chunkを完走し、blocked / exception / identity driftでは停止する。
- overall resultはpopulation比例のraw evidence、typed array、GPU resourceを保持せず、全体first mismatchを最大16件、stage-local representativeを各stage最大4件に制限する。
- 完了済みproduction frameは、実際に使用したasset / SPL4 SHA-256 / record count、build configuration、44 projection values、camera / time / request / production / presentation identityをboundedでdeep-frozenな`productionEvaluationInputContract`としてpublishする。このobserver metadataはparent production readinessを変更しない。
- controllerは明示的one-shot debug APIだけから実行し、production snapshotをstrict preflightする。production deviceを入力として受け取らず、fresh diagnostic adapter / deviceを取得し、同時二重実行を拒否し、全経路でdeviceをcleanupする。RAF、scheduler、captureから自動実行しない。

single-chunk contractは`single-contiguous-resident-chunk-semantic-comparison`
（schema `phase3-population-aligned-semantic-comparison-v5`）、orchestration contractは
`full-production-resident-range-eight-chunk-semantic-comparison`
（schema `phase3-population-aligned-semantic-comparison-orchestration-v4`）、controller contractは
`phase3-population-semantic-comparison-controller-v1`である。

### 13-stage comparison contract

比較順序は次の13 stageで固定する。

1. `temporalEligibility`
2. `conditionalStatePosition`
3. `conditionalWorldCovariance`
4. `cameraSpaceCovariance`
5. `projectionJacobian`
6. `screenCovariance`
7. `conic`
8. `radius`
9. `productionRasterEligibility`
10. `projectedCenter`
11. `cameraDepth`
12. `webgpuInclusivePixelBounds`
13. `normalizedInclusiveTileBounds`

actualの最初の8 stageはfresh diagnostic dispatchのGPU evidenceを使用する。raster companionは、
同じfresh diagnostic device上で生成したnative production tile-input storage bufferをobserver WGSLが
読み、後半5 stageを追加する。これは元のproduction frameと同一dispatchのreadbackではなく、
`actualEvidenceSameProductionDispatch: false`である。CPU expectedはSPL4、applied build configuration、
projection / view contract、CUDA式から独立再構築する。production計算はdiagnostic readbackに依存せず、
`productionCalculationDependsOnDiagnosticReadback: false`を維持する。

temporalまたはraster eligibilityにより後続stageが適用不能なrecordは、missing / invalidではなく
`not-applicable`としてaccountする。N/A-only stageは全record accountingが成立すればcompleteであり、
chunk partition位置でaggregate結果を変えない。stage-local representativeは既存comparison evidenceの
bounded serializationであり、新しいproduction traceや追加readbackではない。

### Fix5 portable pixel boundaryとFix6 tile bounds

Fix5はWGSLで許容されるf32境界差についてraw mismatch、precision-aligned、semantic residualを分離した。
pixel boundsのsrcIndex `803621`, `817028`, `820164`, `833130`はraw 4 records / 4 componentsを保持する。
expected / actual双方のcenter、radius、viewport、bounds dependencyを検証し、actualをexpected生成へ
使わない独立expected envelope内の場合だけ`precision-aligned`とする。pixel toleranceは0のままであり、
global first mismatchにはsemantic residualだけを含める。

Fix5のcanonical baseline artifactは
`/home/demo/work/json/phase3_step121_impl2_fix5_000151_v13_population_semantic_comparison.json`
（SHA-256 `d36b95ce01d4a1c18c05cd53f26814ca338a3b27d211d9be181ffba5e26759a4`）である。
この時点のtile boundsは5,079 records / 5,434 componentsのsemantic residualを持ち、
内訳はminX 0、minY 0、maxX 2,706、maxY 2,728で、すべてmax側がexpectedより1 tile大きかった。
旧productionがinclusive pixel maximumをtileSizeで割っていたのに対し、
CUDA `getRect`はcenter / radiusからfloat-to-intのゼロ方向切り捨てでexclusive maximumを生成することが
原因だった。Fix6はproduction count / scatterとraster observerを同じCUDA-aligned common WGSL helperへ
接続した。min / maxExclusiveをexclusive gridへclampし、nonempty rectだけをinclusive maxへ変換し、
empty rectはloop前に終了する。pixel contract、CPU expected、tolerance、precision classificationは変更していない。

### Accepted Fix6 browser evidence

canonical artifactは
`/home/demo/work/json/phase3_step121_impl2_fix6_000151_v13_population_semantic_comparison.json`、
SHA-256は`db827a6741e4111f57a9469f2dd20535fdbdcee696fe459dd926017899d88a33`である。
固定条件はsource range `[524288, 1048576)`、camera `000151_v13`、frame 151、time 23.2、
1280 x 720、top-left / y-downである。

- controller / orchestration decisionは`match`、blocked reasonsは空、evidenceはcompleteである。
- diagnostic acquisitionは`ready`、cleanupは`destroyed`である。
- 8/8 chunks、524,288 processed / unique records、first srcIndex 524288、last srcIndex 1048575である。
- missing、extra、duplicate、out-of-range、order mismatch、gap、overlapはすべて0である。
- 全13 stageのmissing / invalid / semantic residualは0で、overall `firstMismatches`は空である。
- pixel boundsはraw 4 / 4、precision-aligned 4 / 4、semantic residual 0 / 0、classification `precision-aligned`であり、上記4 representativesを維持する。
- tile boundsはraw / precision-aligned / semantic residualがすべて0 / 0、maximum absolute error 0、classification `match`で、stage-local representativeは空である。

ユーザーの実browser目視ではViewerは正常かつnonblankで、明らかなcoverage hole、黒い欠落、
ちらつきは確認されなかった。Fix6では新しいPNGをcaptureしていないため、Fix5 PNGとのbyte identityや
CUDA Reference PNGとのvisual parityをこのevidenceから主張しない。

### Step121 completion boundary

Step121は、上記固定条件とresident rangeについて、全524,288 source recordsを欠落なく処理し、
portable pixel-boundary classificationを含む定義済み13 stageのrecord-local semantic parityを
実browser / 実WebGPU device上で成立させた限定scopeで完了する。これは全値のbitwise一致ではなく、
pixel boundsの4 raw differencesを明示的に保持したsemantic matchである。

次は未達であり、Step121完了から推論しない。

- Fix6後のfresh WebGPU PNGと同じfixed-range CUDA Reference PNGの定量的visual parity
- opacity / SH / color evaluation parity
- tile-reference population / count、depth key / global sort、per-tile ordered reference listのparity
- compositor alpha accumulationとfinal RGB pixel parity
- Acceptance Level 4
- 全3,231,588 source recordsのfull-scene correctnessとmatched full-scene CUDA comparison
- nonresident recordsがcoverage、ordering、accumulationへ与える影響
- 複数camera / time、interactive camera / time acceptance
- performance / scalability、LOD / streaming、final production acceptance

Step121完了時点の次候補は、Fix6後のfresh production PNGを同条件のCUDA Referenceと再比較し、
差が残れば最初の未比較downstream stageをread-onlyで分類することだった。後続Step122はこの比較と
alpha分類を実施した。現在地はSection 23を正本とする。全3,231,588 recordsをsilent omissionなく
処理するmatched full-scene gateが引き続き必須であることは変わらない。

## 21. Step122 Fixed-Range Downstream Alpha/RGB Semantic Acceptance

Step122は、Step121で閉じた13-stage record-local semantic parityの下流について、fresh
fixed-condition PNG差を定量化し、production tile-input alpha / RGB evidenceを追加して、最初の
downstream mismatchを分類する作業である。Impl1 Fix2の実browser受入とInvestigation7のRGB分類まで
進んだ後、Impl1 Fix3のexpected-oracle C2修正とexpected-only browser gateを完了した。
Investigation8はImpl3前に必要なdiagnostic RGB / alpha validity分離を独立Fix4へ分類し、Impl1 Fix4は
その実装と自動検証を完了した。Step122 Impl3はproduction degree-2 spatial SHを実装し、固定
524,288 recordsの15-stage browser semantic gateをacceptedにした。Validation2はImpl3後のfresh
fixed-condition PNGを比較し、visual error magnitudeの改善と残る未分類差を正式baselineとして確定した。
Step122全体とfixed-condition PNG parityは未完了である。Section 14の
Step119 Investigation7はconditional covarianceのhistorical first mismatchであり、ここでいう
Step122 Investigation7とは別の調査である。

### Validation1: fresh fixed-condition PNG comparison

canonical comparison artifactは
`/home/demo/work/json/phase3_step122_validation1_000151_v13_fixed_visible_comparison.json`
（SHA-256 `c165adfd69a7e36a09869eff4f7045d10583b617ae5e24f9ee70464cd888eb12`）である。
source range `[524288, 1048576)`、camera `000151_v13`、frame 151、time 23.2、1280 x 720、
black background、top-left / y-down、RGB 3-channel比較はreadyで、normal orientationの誤差は
vertical flipより小さい。MAEは`7.347297815393518`、RMSEは`29.717332263813546`、最大絶対誤差は
255、different pixelsは152,988、ratioは`0.16600260416666668`である。旧baselineよりMAE、RMSE、
different pixel countは改善したが、classificationは
`comparison-ready-difference-unclassified`であり、exact visual parityではない。

### Impl1: 15-stage downstream evidence

Impl1 artifactは
`/home/demo/work/json/phase3_step122_impl1_000151_v13_population_semantic_comparison.json`
（SHA-256 `cf3a1fce992b57b6a898cd8ce70ce628870a45bcebd8d8cf3ed65cfedbbdaa4a`）である。
既存13 stageの後へ`productionTileInputAlpha`と`productionTileInputRgb`を追加した。actualはfresh
diagnostic device上のnative production tile-input GPU evidence、expectedはSPL4とCUDA式からの独立
再構築であり、production runtimeはdiagnostic readbackへ依存しない。alphaは3,472 records / 3,472
components、最大誤差`0.049190036767702866`、RGBは21,012 records / 47,469 componentsで、最初の
downstream mismatchはalphaだった。alphaの直接原因はrecord-local `[0.05, 0.99]` clampだった。

### Impl2 and Impl1 Fix1: production alpha semantics and bounded evidence

Impl2 artifactは
`/home/demo/work/json/phase3_step122_impl2_000151_v13_population_semantic_comparison.json`
（SHA-256 `a637f5c6caa9aacc89a4f67cfb8abd027a37125ce1419f328936e153f65ecf3f`）である。
production record-local alphaを`sigmoid(raw opacity logit) * temporal marginal`へ同期し、alpha
clampだけを除去した。temporal eligibility threshold `0.05`とineligible early returnは維持した。
既存13 stageのsemantic residualは0のまま、alpha mismatchは6 records / 6 components、最大誤差
`0.000010596700683596083`へ減少し、RGBは21,012 / 47,469のままである。ユーザーの実browser目視では
Viewer表示は正常だった。Impl2 PNG比較はMAE `7.344609013310185`、RMSE
`29.729211894334124`、different pixels 152,600、ratio `0.1655815972222222`で、小幅な変化に
とどまりvisual parityには達していない。

Impl1 Fix1 artifactは
`/home/demo/work/json/phase3_step122_impl1_fix1_000151_v13_population_semantic_comparison.json`
（79,394 bytes、SHA-256 `6e7149fea4ebc5b83943bafc9bc7573b13f70f151d24fed2e43d76e8eb973ee8`）
である。既存readbackを再利用してalpha stage-local representative上限を8へ広げ、srcIndex
`823750`、`826596`、`828798`、`829562`、`832266`、`870555`の6件を6 / 6、
`truncated: false`で保存した。8 / 8 chunks、524,288 recordsのcoverageはcompleteで、missing、
invalid、duplicate、gap等はない。diagnostic device取得と破棄も正常である。現行working-tree contractは
orchestration schema `phase3-population-aligned-semantic-comparison-orchestration-v5`、alpha
representative上限8、RGB上限1、global上限16、controller上限110,000 bytesである。

### Investigations3-6 and fixed CUDA Reference boundary

Investigation3-5は残る6件のinput、actual、旧JavaScript double expected、f32演算境界を切り分けた。
任意CUDA GPU、CUDA version、driver、compiler、PyTorch kernel、`__expf`実装をすべて包含するportable
CUDA envelopeは保証仕様が不足するためacceptance requirementにしない。比較対象のCUDA Referenceは、
canonical `4dgs310`環境、render-state manifest、asset / SPL4 identity、camera、time、range、生成済み
artifactで固定する。CUDA Reference／PNG比較用Pythonの正本は
`/home/demo/miniconda3/envs/4dgs310/bin/python`である。portability contractは、この再現可能なfixed
CUDA baselineに対するWebGPU / WGSL runtime側へ適用する。

Investigation6はactual GPU値から閾値を逆算せず、W3C WGSLのf32 accuracy、rounding、reassociation、
fusion、subnormal規則をoutward intervalとしてcanonical range全524,288 recordsへ伝播した。eligibleは
21,818 recordsで、nonfinite、subnormal、guard-dependent、eligibility threshold不確定はいずれも0。
最大alpha absolute envelopeは`0.000005513429641723633`（srcIndex `817431`）、最大ULP envelopeは
665 ULP（srcIndex `834848`）だった。tolerance `1e-5`を超えるrecordは0 / 21,818、安全余裕は
`0.000004486570358276368`である。既知6件のactualは全件interval内で、f32 central oracleとの差は
0または1 ULPだった。partition size 65,536、131,072、70,000で結果は不変である。

このintervalはWGSL仕様を安全側へ伝播した保守的外包であり、current GPUの実測誤差や許容結果の
tight supremumではない。結果は固定rangeだけに対する
`candidate-1-ready-under-fixed-cuda-reference`であり、全3,231,588 recordsへの一般化ではない。

### Accepted Impl1 Fix2 browser evidence

Impl1 Fix2のcanonical browser artifactは
`/home/demo/work/json/phase3_step122_impl1_fix2_000151_v13_population_semantic_comparison.json`
（SHA-256 `e4dd6c271603cc1a1436ac9010daf09e468c3b5e5950ff2f86e3f754141554ae`）である。
controller / orchestration decisionは`mismatch`、blocked reasonsは空で、8 / 8 chunks、524,288
processed / unique records、missing 0、invalid stage evidence 0、coverage completeである。upstream
13 stageのsemantic residualは0である。

`productionTileInputAlpha`は21,818 eligible recordsについて0 records / 0 components mismatch、
最大絶対誤差`2.682209014892578e-7`、classification `match`で、expected provenanceは
`phase3-production-tile-input-alpha-f32-central-v1`である。`productionTileInputRgb`は21,012 records /
47,469 components mismatch、最大絶対誤差`0.9877071223567659`で、最初のsemantic / downstream
mismatchである。ユーザーの実browser目視は正常だった。この受入はStep122、PNG parity、Acceptance
Level 4の完了を意味しない。

### Step122 Investigation7: independent RGB problems

Investigation7は固定range全524,288 recordsをread-only replayし、現行production式がartifactのRGB
aggregateとsrcIndex 659525のrepresentativeを再現することを確認した。原因分類3は
「productionとexpected oracleの双方に独立した問題がある」である。

expected oracleはcoefficient-major RGB triplet、`sh[0]`から`sh[8]`、original-position camera
direction、lower-only clampを使用する点ではCUDAと一致する。一方、degree-2 C2の`sh[5]` yz項と
`sh[7]` xz項を正符号としており、CUDAの`[+, -, +, -, +]`と異なる。現行oracleとcanonical oracleは
21,071 records / 48,691 components、最大`0.1586197858`異なる。JS doubleとsource-order f32 replayの
最大差は約`3.36e-7`で、tolerance `1e-5`におけるmismatchは0である。

production evaluatorは`clamp(f_dc + 0.5, 0, 1)`を使用する。CUDA spatial degree-2 contractに必要な
`SH_C0`、degree-1の3項、degree-2の5項、`f_rest`、original-position camera directionがなく、CUDAに
ないupper clampがある。actual RGBはnative production tile-input値をobserverが加工せず保存したもので、
observer差ではない。expected generationはactualへ依存せず、production calculationもdiagnostic
readbackへ依存しない。

Investigation7時点では、expected符号だけをCUDAへ直した場合、現行production actualとの差は
21,030 records / 47,529 components、最大`0.9841542530`、最初はsrcIndex 659525 / Rになると
予測した。これはoracleを正本へ同期した結果であり、productionの悪化を意味しないという受入境界だった。

### Accepted Impl1 Fix3 expected-only browser evidence

Impl1 Fix3はcomparison expected oracleのdegree-2 C2符号だけをcanonical CUDAの
`[+, -, +, -, +]`へ同期し、`sh[5]` yz項と`sh[7]` xz項を負符号へ修正した。single-chunk focused
smokeは、非零のXYZ方向を使うone-hot fixtureで`sh[0]`から`sh[8]`までのcoefficient-major RGB
tripletと各basisの数値結果を固定した。output schema / provenance vocabulary、expected / actual
独立性、production sourceは変更していない。

canonical browser artifactは
`/home/demo/work/json/phase3_step122_impl1_fix3_000151_v13_population_semantic_comparison.json`
（66,699 bytes、compact JSON 44,305 bytes、SHA-256
`af8056194704fe15cf48de08a042c0221f971f2d960c352b6fea280c8176f639`）である。controller /
orchestration decisionは意図どおり`mismatch`、blocked reasonsは空で、controller resultはboundedかつ
JSON serializableである。fresh diagnostic deviceは`ready`で取得され、cleanupは`destroyed` /
completedである。production deviceはinputとして受け取られず使用もされていない。expected生成は
actualへ依存せず、production calculationもdiagnostic readbackへ依存しない。

requested range `[524288, 1048576)`の8 / 8 chunks、524,288 processed / unique srcIndexを完了し、
first / last srcIndexは524,288 / 1,048,575である。missing、extra、duplicate、out-of-range、order
mismatch、gap、overlapはすべて0で、coverage completeである。全15 stageのmissing / invalid countは
0、upstream 13 stageのsemantic residualは0 records / 0 componentsである。既知の
`webgpuInclusivePixelBounds` raw差4件は`precision-aligned`でsemantic residual 0を維持する。

`productionTileInputAlpha`は21,818 recordsを比較し、mismatch / semantic residualともに0 records /
0 components、最大絶対誤差`2.682209014892578e-7`、tolerance `1e-5`、classification `match`である。
`productionTileInputRgb`は21,818 records / 65,454 componentsを比較し、21,030 records / 47,529
componentsのmismatch / semantic residual、最大絶対誤差`0.9841542530436316`、classification
`mismatch`である。最初はsrcIndex 659525のRで、expected `0.048680115709618554`、actual `0`、
absolute error `0.048680115709618554`である。first semantic / downstream mismatchは引き続き
`productionTileInputRgb`である。Fix2の21,012 / 47,469と最大`0.9877071223567659`はhistorical
baseline、Fix3の21,030 / 47,529がcurrent accepted baselineであり、この変化はproduction悪化ではない。

ユーザーの実browser目視は「見た目に支障なし」で、expected-only変更として期待どおりだった。
Fix3ではPNGを保存していないため、PNG identity、fixed-condition visual parity、production RGB parity、
Step122またはAcceptance Level 4の完了を意味しない。

### Step122 Investigation8 and completed Impl1 Fix4 validity separation

Investigation8時点のdiagnostic comparison consumerはnative tile-inputの`colorAlpha`を共有
`colorAlphaValid`で判定し、RGB全成分へfiniteかつ`0 <= value <= 1`、alphaへ`0 < alpha <= 1`を
同時に要求していた。canonical CUDA degree-2 spatial SHはlower-only clampで正当なRGB > 1を生成し得る
ため、当時のconsumerはRGB evidenceだけでなく正常alphaまでinvalidにしていた。この記述はFix4前の
historical defectであり、現在の実装状態ではない。

Impl1 Fix4は`demo/js/webgpu_population_aligned_semantic_comparison.js`で共有predicateを除去した。
`productionTileInputRgb`は3成分すべてfiniteかつ`>= 0`、上限なしでvalidとなる。RGB > 1は通常の
comparison対象で、expectedと一致すればmatch、toleranceを超えればmismatchになる。negativeまたは
nonfinite RGBはRGB stageだけをinvalidとしてfail closedにし、数値error / mismatch aggregateへ混入しない。
`productionTileInputAlpha`はRGBと独立してfiniteかつ`0 < alpha <= 1`を要求する。invalid RGBはalphaを、
invalid alphaはRGBを無効化しない。

変更はcomparison consumerと
`tools/smoke_step121_impl1_population_aligned_single_chunk.mjs`の2ファイルだけである。focused fixtureは、
RGB `[1.25, 0.5, 0.75]`のmatch、expected 1.25 / actual 1.0のvalid mismatch、negative / nonfinite
RGBのRGB-only invalid、alpha 1.25のalpha-only invalidを固定した。negative / nonfinite RGBではRGBが
`blocked-incomplete-evidence`、invalid 1、mismatch 0、max error nullとなる一方、alphaはvalid /
compared 1、classification `match`を維持する。invalid alphaでもRGBは同じくvalid / compared 1、
classification `match`を維持する。

single-chunk smokeは52 casesすべて成功した。変更JS / MJS syntax check、eight-chunk orchestration、
controller、alpha f32 central oracleを含むsingle-chunk回帰、`npm run build`、`git diff --check`、
trailing-whitespace確認も成功した。schema、provenance、stage名 / 順序 / tolerance、first mismatch、
bounded representative、公開return shapeは不変であり、production、observer、orchestrator、controller、
common comparison contract、capture、Summaryは変更していない。したがってFix4は実装・自動検証完了だが、
production RGB semantic変更またはbrowser acceptanceではない。

Fix4完了時点のproductionはRGBを`[0,1]`へclampし、Fix4単独ではactual RGB > 1 branchをbrowserで
生成できなかった。この限定理由によりstandalone browser gateは省略したが、一般のdiagnostic consumer変更へ
browser不要を拡張しない。後続Impl3の15-stage browser gateは実施済みで、更新後production RGBとalphaの
独立evidenceを0 mismatchで確認した。

### Implemented Impl3 RGB contract and fixed-range browser semantic acceptance

Step122 Impl3はproduction evaluatorへdegree-2 spatial SHを実装した。固定CUDA契約は、
`sh[0] = f_dc`、`sh[1..8] = f_rest[0..23]`の
coefficient-major RGB triplet、`SH_C0`、degree-1の3項、degree-2の5項、C2符号
`[+, -, +, -, +]`、original Gaussian positionからcamera world positionへのdirection、最後の`+0.5`、
lower-only clampである。conditional / temporally shifted positionとupper clampは使用しない。

Impl3はversioned common candidate-attribute layoutを正本とし、storage bindingを追加せず
既存binding 4を、`vec0 = (f_dc.rgb, meanScale)`、`vec1 = (scaleXYZ, sourceCode)`の既存配置を維持した
8 vec4 / 32 floats / 128 bytes per recordへ拡張する。
後続6 vec4へ`f_rest[0..23]`を連続packingするため、`f_rest`部分は96 bytes、layout全体は128 bytesで
あり、両者を混同しない。`renderAttributes`、native tile input、raster companionの既存strideは変更しない。
original xyzはbinding 0、camera world positionは既存projection / view contractから再構築する。
layout versionは`phase3-step122-production-candidate-attribute-input-layout-v1`、Gaussian attribute
evaluation contractは`phase3-step122-webgpu-gaussian-attribute-evaluation-contract-v2`である。
spatial degree 3以上とtemporal SHはdeferredで、`fullGaussianAttributeEvaluationInWgsl`はfalseを維持する。

Impl3は`activeShDegree === 2`だけを受理し、`fdcDim >= 3`、`frestDim >= 24`、typed-array長、全係数finite、
全srcIndex範囲、非負整数の`activeShDegreeT`をdispatch前に検査する。fallback zero、部分係数、degree downgradeは禁止し、unsupported
inputはbuffer allocation前にfail closedとする。`f_rest`部分だけなら65,536 / 524,288 /
3,231,588 recordsで6 MiB / 48 MiB / 310,232,448 bytes（約295.86 MiB）である。candidate layout全体は65,536 recordsで8 MiB、固定
524,288 recordsで64 MiB、全3,231,588 recordsで413,643,264 bytes（約394.48 MiB）になる。現行
fixed-range workset readinessは256 bytes / recordのcapacity契約を満たすdevice limitを前提にするため
固定rangeの64 MiBはその境界内だが、diagnostic pathはworkset ownerを通らない。共通layout builderが
実byte lengthを`maxStorageBufferBindingSize`と`maxBufferSize`へ照合し、allocation前にfail closedする。
production storage binding数8と`maxStorageBuffersPerShaderStage >= 8`を維持する。

productionではcandidate inputをsubmit完了後に破棄し、state / render / footprint出力だけをtile input /
compositorまで保持する。diagnosticでは入力・出力・readbackをcall-scopedで破棄する既存resource ownershipを
維持する。production計算はdiagnostic readbackへ依存しない。今回のbrowser bundleはproduction layoutと
device-limit preflightを直接serializeしていないため、実browser device-limit値はinconclusiveである。
約394.48 MiBのfull-scene単一bindingを前提にせず、全45 `f_rest` floats、Step / camera / frame / time /
srcIndex固有分岐も追加しない。

production compositorに残るper-splat RGBの`[0,1]` clamp、sample alphaの`0.98` clamp、`rgba8unorm`
量子化、ordered compositing / final RGBはImpl3とは別の下流責務である。Impl3はrecord-local
`productionTileInputRgb` parityだけを閉じ、これらを変更していない。

Impl3のbrowser acceptance bundleは
`/home/demo/work/json/phase3_step122_impl3_000151_v13_browser_acceptance.json`
（55,471 bytes、SHA-256
`3dd4706918191c46727dbf54d68c201524a6e28e8f690bed9cc24ce990282e4b`）である。固定条件はsource range
`[524288, 1048576)`、camera `000151_v13`、frame 151、view 13、time 23.2、1280 x 720、black
background、top-left / y-downである。ユーザーの実browser目視は「見た目に変化なし、支障なし」だった。

controller / orchestration decisionは`match`、blocked reasonsは空で、fresh diagnostic device acquisitionは
`ready`、cleanupは`destroyed`かつcompletedである。8 / 8 chunks、524,288 processed / unique records、
coverage completeで、missing、extra、duplicate、out-of-range、order mismatch、gap、overlapはすべて0である。
全15 stageにmissing / invalid evidenceはなく、upstream 13-stage semantic residualは0である。
`productionTileInputAlpha`は21,818 recordsを比較してmismatch 0 records / 0 components、最大絶対誤差
`2.682209014892578e-7`、tolerance `1e-5`、classification `match`である。
`productionTileInputRgb`は21,818 records / 65,454 componentsを比較してmismatch / semantic residualとも
0 records / 0 components、最大絶対誤差`3.3570440627350706e-7`、tolerance `1e-5`、classification
`match`である。first semantic / downstream mismatchはいずれもnullである。これによりImpl3の固定
524,288-record browser semantic gateはacceptedとする。

Console側の集約`accepted:false`はsemantic comparison failureではない。capture commandが
`getLastRenderResult()`のWebGPU-exclusive lifecycle wrapperへGaussian attribute contractとcandidate layoutの
direct publicationを期待した一方、そのwrapperは`webgpu-exclusive-frame-lifecycle-pending`としてWebGPUが
Viewer canvas lifecycleを所有しWebGL2 frameを抑止する正常境界だけを公開した。このbundleでは
`gaussianAttributeContractVersion`、`evaluationMode`、`candidateAttributeInputLayout`がnullであるため、
direct serialized production-layout evidenceはinconclusiveである。`nativeProductionFrameDataPathReady`はtrueで、
exact layoutの根拠は現行source、focused smoke、自動検証、およびreadyなproduction data pathである。
確認専用にcapture / Viewerへlayout publicationを追加するFixは不要であり、artifactがlayoutを直接証明したとは
扱わない。

### Step122 Validation2 post-Impl3 fresh PNG baseline

Validation2はsource range `[524288, 1048576)`、524,288 records、camera `000151_v13`、frame 151、
view 13、time 23.2、1280 x 720、black background、top-left / y-down、WebGPU production、fixed
CUDA Reference、RGB 3-channelの固定条件で、Impl3後のfresh production PNGを比較した。ユーザーの
実browser目視はnonblankで表示に問題なく、明らかなruntime failureはなかった。

capture command contractは`ready`、file checkは`OK`で、required filesはすべて存在し、JSONはすべて
validである。PNG statusは`success`、PNGは1280 x 720 RGBA、nonblankで、SHA-256は
`9fbacdaa328489494e44e87f8cb3e29d1ed435438174cd53684fa4c340f6793f`、RGB nonblack pixel countは
153,916である。capture freshnessはknown、stale detectionはfalseで、presented frameとrequested stateの
双方に一致する。production capture expectationとruntime healthはreadyで、WebGPU validation error、
Invalid CommandBuffer、queue submit failure、device lostはない。required / written / scattered / sorted /
composited reference countsはすべて229,328である。generic visible-record dry-runはstatus `ok`のcapture
health補助証拠であり、production 524,288-record semantic controllerまたはImpl3 15-stage gateの再実行証拠
ではない。

current canonical comparison artifactは
`/home/demo/work/json/phase3_step122_validation2_000151_v13_fixed_visible_comparison.json`である。
comparison conditionはready、classificationは`comparison-ready-difference-unclassified`で、CUDA PNG
SHA-256は`fc0cc07de5f708300e6971fcb35803fe9d9dd3bbf4643f86a677e31f540bcee9`、WebGPU PNG
SHA-256は上記`9fbacdaa...f6793f`、same SHA-256はfalse、same sizeはtrueである。CUDA / WebGPU
nonblack pixelsは131,093 / 153,916。MAEは`6.42220630787037`、RMSEは`27.03804631164564`、
maximum absolute errorは254、any-RGB-channel different pixelsは153,894、ratioは
`0.16698567708333334`である。normal MAE `6.42220630787037`はvertical-flip MAE
`14.722511574074074`より小さく、lower-error orientationは`normal-webgpu-png`である。上下反転を
残差原因として扱わない。

Validation1のMAE `7.347297815393518`、RMSE `29.717332263813546`、maximum error 255、different
pixels 152,988、ratio `0.16600260416666668`、WebGPU nonblack pixels 152,737に対し、Validation2は
MAE `-0.9250915075231481`（約12.6%改善）、RMSE `-2.6792859521679055`（約9.0%改善）、maximum
error `-1`である。一方、different pixelsは`+906`、ratioは`+0.0009830729166666607`、WebGPU
nonblack pixelsは`+1,179`である。Impl3 degree-2 spatial SHはvisual error magnitudeを明確に改善したが、
差分pixel populationを減らしておらず、exact parityまたは「ほぼ一致」を意味しない。

このPNG差は、WebGPU actualとcurrent expected oracleの固定524,288-record 15-stage internal matchを
否定しない。Validation2時点では差を15-stageより下流と仮置きし、Investigation9でproductionとCUDAの
tile-reference population / countを最初に確認する計画だった。後続Investigation10は、その前提より上流の
CUDA camera-to-rasterizer handoffでscreen / footprint contractが分裂していることを確定した。現在の
Investigation10の原因分類と当時の完了境界はSection 22、現在のfreeze境界と依存順はSection 23を正本とする。

CUDA manifestはtemporal SH degree 0を記録する一方、checkpoint restore後のruntime active temporal
degreeは2である。ただしCUDA temporal SH処理はspatial degree > 2の内側にあり、今回のspatial degree 2
ではtemporal coefficientの寄与は0で、このprovenance gapは今回のRGB分類を覆さない。
`convert_SHs_python=false`と`compute_cov3D_python=false`はcall pathから確定したが、manifestは両最終値を
直接保存していない。manifestのCUDA revisionは`b8712f36c68140dd105f4f022b02fed4466bcabb`、render時
repositoryはdirtyであり、現在のclean checkout `7663366f823b0beea9cedf76013840f20f7cd563`をrender時の
完全source identityとして扱わない。committed `auxiliary.h`のC2負符号との一致は補助根拠であり、
dirty patchの完全保存を意味しない。spatial degree > 2、temporal SH、CUDA再生成、full-scene一般化を
受け入れる前にこのprovenance gapを解消する必要がある。

### Validation2時点のremaining gates and dependency order

次はValidation2時点の履歴的な依存順であり、現在順はSection 23を正本とする。

1. **完了済み**: Step122 Impl1 Fix3でexpected oracleのC2符号だけをCUDA正本へ同期する。
2. **完了済み**: production sourceを変更せず、Fix3の15-stage browser comparisonで固定524,288件のcorrected baselineを独立して受け入れる。
3. **完了済み**: Step122 Investigation8でdiagnostic RGB / alpha validityの独立不具合を分類する。
4. **完了済み**: Step122 Impl1 Fix4でvalidityを分離し、52-case focused smokeと関連回帰を完了する。Fix4の非干渉はImpl3 browser gateで確認する。
5. **完了済み**: Step122 Impl3でproduction degree-2 spatial SHをversioned candidate-aligned resource contractへ実装し、自動検証を完了する。
6. **完了済み**: Impl3の15-stage browser comparisonでproduction RGB / alpha semantic residual 0、完全coverage、fresh diagnostic-device lifecycleを確認する。
7. **完了済み**: Step122 Validation2でpost-Impl3 fresh fixed-condition PNGを同じCUDA Referenceと比較し、MAE / RMSE改善と残る未分類差を記録する。
8. **当時の次候補（Investigation10により保留）**: productionとCUDAのtile-reference population / countを比較し、一致した場合だけdepth key / sort、per-tile ordered reference list、compositor accumulation、final RGB / PNG encodingへ順に進む。
9. 固定rangeとは独立して、全3,231,588 recordsをsilent omissionなく処理するmatched full-scene gateを完了する。

Fix3 expected-only gateはalpha 0 / 0、upstream 13-stage semantic residual 0、RGB 21,030 records /
47,529 components、最大`0.9841542530436316`、最初はsrcIndex 659525 / Rとしてacceptedになった。
Impl3 gateはproduction sourceをdegree-2 spatial SHへ変更した後、alpha / RGBとも0 mismatch、first semantic /
downstream mismatch nullとしてacceptedになった。Validation2はImpl3後のfresh PNGを比較し、MAE / RMSEを
改善したが、classificationは`comparison-ready-difference-unclassified`のままである。

次は未達であり、Step122やAcceptance Level 4の完了を主張しない。加えて、Investigation10により
CUDA-direct camera clamp / Jacobian / screen covariance / conic / radius / tile parityの再検証が先行gateとなった。

- fixed-condition PNG visual parity、opacity / SH / color全体のparity
- tile-reference populationの最終定量確認、depth key / sort、ordered per-tile reference listのparity
- compositor accumulationとfinal RGB parity
- Acceptance Level 4
- 全3,231,588 recordsのmatched full-scene correctness
- interactive camera / time acceptance
- performance / scalability、LOD / streaming、final production acceptance

## 22. Step122 Investigation10 CUDA Reference Camera/Raster Contract Reclassification

このSectionは、Step121 / Step122の既存実装・browser evidenceを取り消さず、13 / 15-stage gateの
CUDA-direct scopeとValidation2の位置付けを限定する最新の正本である。Investigation10はsourceと既存
artifactだけをread-onlyで照合し、production、CUDA、tool、test、artifactを変更していない。

### Source and artifact lineage

canonical CUDA Referenceは次の固定条件を持つ。

- output directory:
  `/home/demo/work/outputs/sph_scene_4dgs/cuda_reference_step119_impl7_population_0524288_1048576/iter_012000`
- run id: `cuda-reference-000151_v13-20260819T042655Z-80be91a3`
- source range: `[524288, 1048576)`、524,288 records
- camera `000151_v13`、frame 151、view 13、time 23.2、1280 x 720
- manifest camera publication: `fx = fy = 1777.7777777777778`、`tanFovX = 0.36`、
  `tanFovY = 0.2025`、positive FoV

CUDA source lineageでは、`scene/dataset_readers.py`がintrinsics cameraへ有効な`fl_x/fl_y/cx/cy`と
`FoVx = FoVy = -1.0`を設定する。`scene/cameras.py`は`cx > 0`の場合にfocal intrinsicsからfull
projectionを構築する。一方、`gaussian_renderer/__init__.py`はcamera種別を分けず
`tan(FoV / 2)`をrasterizer settingsへ渡し、`rasterizer_impl.cu`はそのtanFovからfocalを再計算する。
`forward.cu::computeCov2D`は同じtanFovでcamera-space x/yをclampし、そのfocalでJacobianを構築する。

この実行でfootprint側へ実際に渡った値は、`tanFovX = tanFovY = -0.5463024974`、
`focalX = -1171.512085`、`focalY = -658.975586`である。manifest builderは`FoV <= 0`を
`fl_x/fl_y`からpositive effective intrinsicsへ正規化して公開したため、canonical artifactは次の
split contractを持つ。

- projected screen center: valid `fl_x/fl_y/cx/cy`を使うfull projection
- footprint clamp / Jacobian / covariance / radius: `FoV=-1` sentinel由来のnegative tanFov / focal
- manifest publication: positive effective focal / tanFov / FoV

### Canonical-eight comparison

canonical srcIndexは
`658947, 771007, 788034, 826401, 835183, 852955, 863505, 906711`である。
CUDA direct evidenceとcurrent expected replayのstage結果は次のとおりである。

| stage | canonical-eight result |
| --- | --- |
| conditional world covariance | 8 / 8 match |
| camera-space position, unclamped | 8 / 8 match |
| camera depth | 8 / 8 match |
| camera-space covariance | 8 / 8 match |
| projected screen center | 8 / 8 match |
| camera-space position, clamped | 8 / 8 mismatch |
| projection Jacobian | 8 / 8 mismatch |
| screen covariance | 8 / 8 mismatch |
| conic | 8 / 8 mismatch |
| radius | 8 / 8 mismatch |
| tile rect | 6 / 8 mismatch |
| `tilesTouched` | 6 / 8 mismatch |

source-levelで最初に異なるのはcamera-space clampである。srcIndex 658947のclamped xはCUDA
`-60.285377502441406`、current expected `14.475767772032562`、absolute error
`74.76114527447396`である。現行15-stage vocabularyで最初に公開される不一致は同recordの
projection Jacobian `j00`で、CUDA `-13.801024436950684`、current expected
`20.943151606426603`、absolute error `34.74417604337729`である。差は

`camera clamp -> Jacobian -> screen covariance -> determinant/conic -> radius -> tile rect -> tilesTouched`

の順で伝播する。CUDA directの`tilesTouched`は`[4,6,16,30,4,6,2,6]`、current expectedは
`[4,16,25,49,4,12,4,12]`である。

### Cause classification and completion boundary

原因分類は`mixed-divergence`である。CUDA executionはraw `FoV=-1` sentinelをfootprint
rasterizerへ流し、manifest publicationは同じcameraをpositive effective intrinsicsとして公開する。
WebGPU expected / productionは公開されたpositive projection contractを使って相互に一致する。この差は
単なるf32 precision differenceでも、WebGPU productionだけの不具合でもない。

Step121では、single-chunk producer、eight-chunk orchestrator、one-shot controller、524,288-record
coverage、fresh diagnostic-device分離、bounded evidence、およびWebGPU actual対current expectedの
13-stage internal matchを完了済みとして維持する。Step122でもdiagnostic Fix群、central alpha f32
oracle、degree-2 spatial SH production、WebGPU actual対current expectedのalpha / RGB match、完全coverage、
device分離、およびValidation2の実測値を維持する。ただし両Stepの13 / 15-stage matchをCUDA-direct
camera clamp、Jacobian、screen covariance、conic、radius、tile bounds、tile-reference parityの証明には
使わない。coherent CUDA Referenceに対するscreen / raster parityは再検証待ちである。Step122 overall、
fixed PNG parity、Acceptance Level 4、full-scene correctnessは未完了である。

### Validation2 historical measurement

Validation2は実行済みであり、artifactを無効または未実行として扱わない。比較値はMAE
`6.42220630787037`、RMSE `27.03804631164564`、maximum absolute error 254、different pixels
153,894、different pixel ratio `0.16698567708333334`、WebGPU production reference count 229,328である。
これは当時実際に生成されたCUDA PNGとWebGPU PNGのhistorical measurementとして有効だが、CUDA側の
camera / raster intrinsicsがsplitしていたため、coherent camera/raster contractとの最終parity baselineでは
ない。修正後のCUDA Referenceで再比較する必要がある。

### Investigation10-time dependency order, superseded by Investigation11

The following list is retained as the historical Investigation10 disposition. It was superseded before implementation by the corrected-baseline decision in Section 23.

1. **完了済み**: Investigation10の原因・完了境界・依存順を正本文書へ同期する。
2. **historical candidate, not executed and no longer current**: intrinsics cameraのCUDA Reference camera-to-rasterizer handoffを修正する。
3. intrinsics cameraとlegacy FoV-only cameraをfocused testで検証する。
4. 実装報告を相談役がreviewする。
5. `/home/demo/miniconda3/envs/4dgs310/bin/python`を使い、既存outputを削除・上書きせず、新しいdirectoryへCUDA Referenceを生成する。
6. canonical eightのcamera clamp、Jacobian、screen covariance、conic、radius、tile rect、`tilesTouched`を再比較する。
7. canonical eightがreadyの場合だけ、固定524,288件のsemantic、tile-reference、PNG比較へ進む。
8. coherent Reference確立後、CUDA `num_rendered`とtile-reference population evidenceのpublicationを別責務として再検討する。
9. fixed-range downstream parity後、全3,231,588件をsilent omissionなく処理するmatched full-scene gateへ進む。

次候補ではintrinsics cameraのraw sentinelをrasterizer settingとして使用せず、`fl_x/fl_y`、width、heightから
effective tanFov / focalを一意に構築し、full projectionとfootprint rasterizerが同じeffective intrinsicsを
共有する必要がある。legacy FoV-only pathは維持し、manifestはraw camera inputと実際にrasterizerへ渡した
effective settingsを区別して保存する。これは設計境界の記録であり、実装方法、変更file、helper配置、
schema変更を認可または固定しない。

`num_rendered`総数だけではper-record `tilesTouched`や`(tile, Gaussian)` reference populationの一致を
証明できないため、camera handoffより先にpublicationへ進まない。direct evidenceのprovenanceには、JSON内
`artifact.sha256`の`5ec5a0af...`が最終bytesではなくpre-enrichment hashである一方、実file SHA-256は
`cbde9927bb625c228b0a119b1784a24b55c6592f43faf04c78e96828e6c6e2f5`である留保がある。また、現行
`ndc` fieldは実際には`T = W * J`の先頭成分、`clip` fieldは
`[focalX, focalY, tanFovX, tanFovY]`である。これらのrenameやfinal-bytes hash publicationはcamera
handoffの直接原因と分離し、必要性とschema影響を別責務として判断する。

## 23. Step122 Investigation11 Corrected-Baseline Freeze and Retraining Track

### Investigation10からInvestigation11への時系列と判断

Investigation10は、intrinsics cameraのfull projectionが正の`fl_x/fl_y/cx/cy`を使用する一方、
legacy CUDA footprint rasterizerがraw `FoVx = FoVy = -1` sentinelからnegative tanFov / focalを
構築していたsplit contractを確定した。Investigation11はtraining sourceと保存artifactのlineageを
追加確認した。既存artifactだけでは、各training iterationで実行されたrenderer bytesとruntime
settingsを暗号学的に一意に証明できないため、formal classificationは
`D: training-provenance-insufficient`である。ただし、legacy forkのsource、training output時刻、
checkpoint/config/camera identities、およびtraining loopが同じrenderer outputをloss、gradient、
visibility、radii、densification、clone/split/pruneへ使用する構造は、legacy checkpointがhistorical
negative footprint semanticsの影響下で最適化されたことを強く示す。

このためユーザーは、historical negative-sentinel semanticsをViewerへ互換実装する案を採用せず、
shared 4DGS rendererのcamera handoffを修正し、corrected rendererで初期状態から再学習する
corrected-baseline trackを選択した。ViewerはStep122途中で凍結する。この節は判断と依存順を記録する
ものであり、camera handoff Fix、diagnostic render、training、export、CUDA Reference生成、または
Viewerの新しいFixを開始・認可しない。

### Immutable legacy historical baseline

Legacy repository identityは次のとおりである。

- CUDA/training fork: `misshiki2-arch/4d-gaussian-splatting-sph`
- legacy CUDA/training HEAD: `7663366f823b0beea9cedf76013840f20f7cd563`
- official upstream base: `63725f21d4adc29669e565ae10e6b3ad6e0d1250`
- Viewer branch: `phase3-webgpu-compute-prototype`
- Viewer HEAD: `c137ae42abbd703994f26ac2c2326d2349021654`

Legacy dataset identityは次のとおりである。

- `transforms_train.json`: SHA-256
  `6c29326bc590774b534f0d334e5fd03f841101cfb23dced5dd18bcc2a13a6068`、5,146 frames
- `transforms_test.json`: SHA-256
  `c8e3ab9e45fa2e80d885a4610a78c77b8c813cda45c6a1b63c6ca5509e9c81fa`、166 frames
- `eval=False` combined camera count: 5,312
- camera intrinsics: `fl_x = fl_y = 1777.7777777777778`、`cx = 640`、`cy = 360`、
  1280 x 720

Legacy training output `/home/demo/work/outputs/sph_scene_4dgs`は次のidentityを持つ。

- `cfg_args`: SHA-256
  `7cbfc1a967be13bca1709bd74e475b73283d0babf5f27fd9c515f9fff9be5499`
- `cameras.json`: SHA-256
  `98193b7ae6b8c0da4b36b757b243c1492edae98cbe937173f5c265a41ee282a3`
- `chkpnt_best.pth`: SHA-256
  `bd44500c3474ef67cd4fe44f26a2c83b6953e30315d07220769002b61b74dcb4`、
  2,572,356,878 bytes、iteration 12,000、3,231,588 records、
  mtime 2026-03-27 13:30:15 JST

このcheckpoint、既存SPL4、CUDA Reference、Step119-122 JSON/PNG、およびValidation2 metricsは
legacy historical baselineとして保存する。削除、rename、move、上書き、またはcorrected output先としての
再利用は禁止する。Validation2のMAE `6.42220630787037`、RMSE `27.03804631164564`、maximum
absolute error 254、153,894 different pixels、ratio `0.16698567708333334`、production reference
count 229,328は実行済みhistorical measurementであるが、corrected baseline acceptanceへは継承しない。

### Confirmed camera handoff defect and corrected contract

Intrinsics camera inputは有効な`fl_x/fl_y/cx/cy`とraw sentinel `FoVx = FoVy = -1`を持つ。
Full projectionはpositive focal intrinsicsを使用する一方、legacy rendererはcamera modeを区別せず
`tan(FoV / 2)`をrasterizerへ渡した。このためlegacy executionは
`tanFovX = tanFovY = -0.5463024974`、`focalX = -1171.512085`、
`focalY = -658.975586`を使用した。Corrected intrinsics contractは
`tanFovX = 0.36`、`tanFovY = 0.2025`、
`focalX = focalY = 1777.7777777777778`である。Camera-space clampからJacobian、screen
covariance、conic、radius、tile bounds、`tilesTouched`までがこの差の影響を受ける。

Checkpointとrenderer semanticsは独立ではない。RendererのRGB/alpha outputはlossとbackward
gradientを決め、visibility/radiiはdensification statisticsとclone/split/pruneを決める。したがって
corrected rendererを正式baselineにする場合、legacy checkpointのcorrected-renderer renderは診断比較に
限定し、正式checkpointは初期状態からのretrainingで生成する。

### Viewer freeze boundary

凍結中も次の完了成果を維持する。

- Step118 native WebGPU production canvas presentationとresident workset/range infrastructure
- Step120 conditional 4D covariance
- Step121 single-chunk producer、sequential eight-chunk orchestrator、fresh diagnostic-device分離、
  bounded comparison
- Step122 central alpha f32 oracle、degree-2 spatial SH、candidate input layout
- Step117以降のgeneration/writer/currentTexture/last-valid ownershipとbrowser/capture/PNG infrastructure

これらは削除または全面失敗扱いにしない。状態は
`infrastructure completed / legacy-baseline acceptance retained / corrected-baseline revalidation pending`
である。凍結中はhistorical CUDA footprintへViewerを合わせる実装、新しいViewer Fix、tile-reference、
sort、compositor、229,328 reference count追跡、current PNG metrics改善、full-scene、performance、LODを
進めない。

Viewer凍結解除には、corrected handoff implementationのreview、intrinsics cameraとlegacy FoV-only
cameraのfocused validation acceptance、from-scratch retraining完了、新checkpoint/SPL4/corrected CUDA
Reference identityの確定、およびnew population countとfixed-range baselineの設計がすべて必要である。

### Corrected retraining dependency order

1. Investigation11の文書同期を完了する。
2. ユーザーがViewer/CUDA repositoryのGit checkpointを作成する。
3. CUDA/training repositoryでcamera intrinsics handoff Fixを一責務として実装する。
4. Intrinsics cameraとlegacy FoV cameraをfocused testで検証する。
5. Manifestへraw camera inputと実際のeffective rasterizer intrinsicsを区別して保存する。
6. Legacy checkpointをcorrected rendererで診断renderする。
7. 診断結果を比較するが、legacy checkpointをcorrected formal checkpointへ昇格しない。
8. 新しい非上書きoutput directoryで初期状態から再学習する。
9. GT、legacy checkpoint + legacy renderer、legacy checkpoint + corrected renderer、new checkpoint +
   corrected rendererを同一条件で比較する。
10. New checkpointからnew SPL4をexportする。
11. Corrected CUDA Referenceを新規directoryへ生成する。
12. New population provenance、record count、camera、timeを確定する。
13. Viewer凍結を解除する。
14. Canonical representative comparisonを行う。
15. New fixed-range semantic comparisonを行う。
16. Tile-reference、sort、compositor、PNG comparisonを行う。
17. New population countを正本とするcorrected full-scene gateを行う。
18. Interactive camera/time、performance、scalability、LODへ進む。

Retraining outputの現時点の候補は
`/home/demo/work/outputs/sph_scene_4dgs_corrected_intrinsics_v1`であるが、未作成・未確定であり、
後続の実行命令で再確認する。New checkpointのrecord countは未確定である。Legacyの3,231,588件、
fixed range `[524288, 1048576)`、canonical representatives、semantic thresholds、reference count、
PNG metricsを新baselineへ自動継承してはならない。

### Completion boundary after synchronization

Camera handoff defectの原因分類、training dependencyのread-only調査、corrected baselineを選択した
ユーザー判断、Viewer infrastructure、およびlegacy browser/artifact取得は完了済みである。Camera
handoff Fix、focused validation、corrected diagnostic render、from-scratch retraining、new checkpoint、
new SPL4、corrected CUDA Reference、new fixed-range parity、corrected PNG parity、Viewer凍結解除、
Acceptance Level 4、corrected full-scene correctness、interactive/performance/LODは未完了である。
