# Phase 3 Current State

更新基準: 2026-08-20 / Phase 3 Step119 Investigation7分類完了時点
対象ブランチ: `phase3-webgpu-compute-prototype`
基準コミット: `156a138 Phase3 Step118: add native WebGPU production frame data path`

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

Step118 完了時点は Level 3 である。

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

Step119のfixed-visible比較、first-mismatch分類、既知evidence fieldの意味分類は完了した。残る作業は、原則として次の順序を守る。

1. Step120で、CUDA-aligned conditional 4D-to-3D covarianceをproduction footprintへ接続する一責務だけを修正する。
2. Step120修正後、camera-space covarianceから後段比較を再開し、次の最初のmismatchで停止する。
3. fixed camera / fixed timeでCUDA reference semantic / visual parityを段階的に進める。
4. 3,231,588 recordsのfull-scene correctness gateを設け、device limit内のfull residency、または全source populationを欠落なく扱うcorrectness-preservingなchunk / streamingでCUDA full-scene Referenceと比較する。
5. interactive camera / timeで同じsemanticsが維持されることを確認する。
6. scene規模、resource lifecycle、performance、scalabilityを確認する。
7. acceptedなparity boundaryの内側でperformance改善を行い、その後early terminationやLODなどの近似・高速化を個別のacceptance contractの下で進める。

原因未確定の段階で、SH、sort、camera、projection、scheduler、retry、RAF などを原因として決め打ちしてはならない。

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

最初の不一致は、4D Gaussianからconditional 3D covarianceをproduction footprintへ渡す境界である。CUDA production rasterizerは左右quaternionとXYZT scaleから4D covarianceを生成し、`Sigma11 - Sigma12 Sigma12^T / Sigma_tt`を使用する。現行WebGPUはtemporal meanでは左右quaternionとscaleTを使用する一方、production footprint covarianceでは左quaternionとscaleXYZだけから通常の3D covarianceを生成している。

captured inputからCUDA式を再構成したconditional 3D covarianceは8/8でCUDA direct値へ`1e-5` tolerance内で一致し、現行WebGPU qL-only式は8/8で同toleranceを超えた。CUDA covarianceはproduction rasterizer debug rowのdirect evidenceである一方、WebGPU conditional world covariance自体は現行artifactへactual GPU intermediateとして直接保存されておらず、captured production inputと現行WGSL式から再構成した。このevidence provenanceをdirect GPU evidenceとして扱わない。

first-mismatch停止条件により、camera-space covariance、projection Jacobian、screen covariance、conic、radius / footprint、opacity / SH color、tile coverage、depth sort、compositor accumulationは未判定である。これらを一致済みまたは問題なしとは主張しない。

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

### Step120

次のproduction実装は、既存のCUDA-aligned 4D rotation / covariance vocabularyを利用してconditional 3D covarianceをproduction footprintへ接続する一責務に限定する。camera transform、projection Jacobian、Y orientation、conic、radius、tile coverage、sort、compositor、opacity、SH/colorを同時に変更しない。修正後はcamera-space covarianceから比較を再開し、次の最初のmismatchで停止する。Step120だけでvisual parity成功を予定事項として断定しない。

## 18. Step119 の到達点と非対象

Step119はfixed-population semantic controlとfirst-mismatch classificationの責務として閉じる。これはfixed-range visual parityまたはAcceptance Level 4の達成を意味しない。

完了済み:

- fixed visible comparison artifactが保存されている。
- CUDAとWebGPUでpopulation、camera、time、resolution、orientationが一致している。
- runtime / capture failureとrendering differenceが分離されている。
- screen centerより上流の一致と、最初のconditional covariance mismatchが分類されている。
- `canonicalExecution.ready`がzero-reference専用fieldであり、nonzero-reference runtime completionと矛盾しないことが分類されている。

Step119後も未達:

- conditional 4D-to-3D covarianceのproduction実装
- conditional covarianceより後段のsemantic comparison
- fixed-range visual parityとAcceptance Level 4
- 3,231,588 recordsのfull-scene correctnessとmatched full-scene CUDA comparison
- final production acceptance

Step119ではcamera、projection、temporal、tile、sort、compositorの変更を行わない。scene全体residency、streaming / LOD、interactive camera / time、performance、retry / RAF / heartbeat、大量trace、Step / camera / frame / time依存のproduction分岐も非対象である。最初のmismatchに対する実装修正はStep120で一責務として扱う。

この非対象指定はStep119 Investigation7へfull-scene実装を混ぜないためのものであり、最終受入れから除外する意味ではない。52万件のfixed-range semantic parityを閉じた後、3,231,588 recordsのfull-scene correctnessを独立した必須gateとして設計・検証する。
