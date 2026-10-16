# Mirror-direction Attention — hypothesis + computational sanity screen (2026-10-16)

**Status: IDEA INTAKE / CPU FORWARD MECHANISM ONLY. No trained model, no NLL comparison, no Mirror-specific capacity or training-efficiency claim.**

## H — 検証する問い

独立した複数Transformer/Mirror Viewに個別のトークンAttentionを実行する代わりに、**共有した因果Token Attentionを1回だけ実行し、同一トークン内の小さいMirror方向コードに対してAttentionする**ことで、K個の有用な論理方向を低い推論/学習コストで相互利用できるか。方向間の協調勾配によって少量学習の収束が改善するかは本pilotでは未測定。

## 最初の構成

基底 \(h_t\in\mathbb R^d\) に共有トークンAttentionを施した \(b_t\in\mathbb R^d\) を用いる。方向 \(i=1,\ldots,K\) に対する \(z_{ti}\in\mathbb R^r\) を小さい共通写像 \(U\in\mathbb R^{d\times r}\) と方向コード \(m_i\in\mathbb R^r\) から生成。簡易例：

\[
 b_t={\rm CausalAttention}(h_{\le t}), \quad z_{ti}=(U^\top b_t)\odot m_i, \\
 \beta_{tij}=\operatorname{softmax}_{j}\left(\frac{z_{ti}^{\mathsf T}z_{tj}}{\sqrt r}\right),\quad
 y_{ti}=b_t+U\sum_j\beta_{tij}z_{tj}.
\]

変数: \(t\in\{1,\ldots,T\}\) は因果トークン番号、\(i,j\in\{1,\ldots,K\}\) はMirror方向、\(d\) は親隠れ次元、\(r\) は方向コード次元、\(\odot\) は要素積、\(\beta\) は方向間の正規化注意重み。\(b_t,h_t,y_{ti}\) は無次元 \(\mathbb R^d\) ベクトル、\(z_{ti},m_i\) は無次元 \(\mathbb R^r\)、\(U\) は無次元の行列、\(\beta\) は \([0,1]\) の無次元係数。全て数値を正規化した合成モデルとしてSI単位は1。式の和と積は次元整合する。実際のMirror変換 \(Q(m_i)\) と単純なゲートを比較するまでこのPilotをMirror固有の証拠にしない。

**Important:** この \(z_{ti}\) は通常の低ランクgain Viewそのものである。Pilotは低コストな軸間Attentionの実行可能性のみを検査し、性能上の新規性は確立しない。\(m_i\) を全方向同一にすると \(y_{ti}\) も同一になり、多数の論理方向は作れていない。学習実験ではMirror対同容量の直接gain/FiLM/talking-heads/native head mixing を比較する。

## T — CPU演算・因果性・交差勾配のPilot

PyTorch 2.10.0+cpu, BLAS 1 thread, CPU 1 thread, float32, batch=2, hidden d=64, K=4, r=16。token長 T=128/256/512。各条件 warm-up 20, 35計測×4群の群内中央値、その中央値を記録。GPUなし、非融合、ランダム固定写像 U と方向コードを使用、モデルの学習はなし。

4構成を比較。1) 共有token SDPAを1回; 2) 4系統の独立token SDPA; 3) 共有token SDPA＋各tokenの4方向に対する低次元SDPA; 4) (T×K)個の位置に対する全結合因果SDPA。未来トークンの入力を変えても前方の出力が変化しないか、方向0の損失から方向1..3のコードに勾配が届くか、全方向コード一致時に全方向出力が一致するかをテスト。

### CPU所要時間(ms)

| T | 共有Token1回 | 独立Token4回 | 共有Token+方向Attention | T×K全結合Attention |
|---:|---:|---:|---:|---:|
|128|0.1320|0.5757|0.4285|2.6296|
|256|0.5155|2.1581|0.9089|9.4211|
|512|1.9240|7.0998|2.5099|34.6184|

共有Token+方向Attentionは独立Token4回に対しT=512で約2.83倍高速、共有Token1回より約1.30倍遅い。CPU PyTorch実装だけの結果であり、Attention入力/出力投影・ffn・decoder・KVキャッシュ・学習時間を含めず、**関数品質も等価でない**。融合GPUや長文での挙動は未知。

計算量の主項（投影・変換費用を除外）：独立K方向のトークンAttention \(O(KT^2d)\)、全結合2軸 \(O(K^2T^2d)\)、共有Token+方向Attention \(O(T^2d+TK^2r)\)。必要な方向状態の生成/読出しにも費用がある。現実の高速化は実測で判定する。

**Causal correctness:** 同時刻t内のMirror方向Attentionには未来トークンが存在しない。因果TokenAttention側の未来禁止マスクだけで未来リークしない。ただし複数トークンを一度に出力するケースでは未知の未来出力をview latentへ混入しないよう別監査が必要。

## D — このpilotの判定

- 性能が等価である保証なしの**計算機構としてFEASIBLE**。テスト4件PASS（別途ベンチ内assert）、各方向コードへの交差勾配と因果性を検証した。
- 単一通常Attentionより速いとする主張は**FAIL**（むしろ遅い）。K個の独立Attentionより高速な条件を確認。
- Mirror固有のNLL、少更新での到達精度、シード分散、一般圧縮・GPU速度は**UNTESTED**。

## C — 強い対立仮説と失敗境界

(1) 改善は単なるhead-mixing/talking-headsやFiLMと同等。 (2) 共有TokenAttentionはview別のtoken-token選択を失い、精度を犠牲にしただけ。 (3) 二軸Attentionで全Viewが同一化し、専門性を失う。 (4) 1回のtoken AttentionでもK個のMirror表現生成/FFN計算が重くなる。 (5) ソフトmax下でK/Vのview変換は一般に共有cacheへ後付け移行できない。 (6) queryとkeyを同じ直交変換で回すと内積は不変であり、Attention多様性がゼロになる。

## U — 重要な不確かさ

CPUマイクロベンチはホストのクロック変動/dispatcher/allocator影響を含む。比較演算の表現力が違い、同品質条件ではない。算術計算量の式は主項のみ。真正の有効性には学習の比較が必要。

## 次の凍結する実験（NOT RUN）

1. 同じ親共有Transformerで **shared-only、native head-mixing/talking-heads、FiLM/lowrank、独立view注意、共有Token+Mirror方向注意、axial token×Mirror注意** を比較。
2. Mirrorとの整合教師/独立教師/異なる周回役割/自然言語の4条件。学習率・データ・updates・tokens・FLOPs・実保存byte・CPU/GPU壁時間の軸を分け、test NLL、worst-view、seed分散と収束時点を測定。
3. 各方向から他方向への関数移転を遮断するアブレーション、同一方向コードcollapse、学習初期のgradient norms/cosines、入出力のhead diversity/有用性、正規化前後を監査。
4. 共有prefix+direction Attentionの効果が強いnative対照に対し有意義な品質/byte/compute-Pareto改善を示すまでMirror固有の容量倍率・高速学習とは言わない。

## Direct prior art

- Axial Attention in Multidimensional Transformers (Ho et al., arXiv:1912.12180), https://arxiv.org/abs/1912.12180 — multi-axis attention.
- Talking-Heads Attention (Shazeer et al., arXiv:2003.02436), https://arxiv.org/abs/2003.02436 — linear head-axis mixing before and after softmax.
- Multi-Head Attention: Collaborate Instead of Concatenate (Cordonnier et al., arXiv:2006.16362), https://arxiv.org/abs/2006.16362 — attention projection sharing.
- Hybrid Routing for a Mixture of LoRA Experts (Huang et al., AAAI 2026), https://ojs.aaai.org/index.php/AAAI/article/view/40383 — cross-expert attention in routing.
- Multi-Head Attention as a Source of Catastrophic Forgetting in MoE Transformers (Chen et al., ICML 2026), https://proceedings.mlr.press/v306/chen26aw.html — head-wise feature routing and gradient collision mechanisms.
- MA-041 https://github.com/Unjuno/functional-quotient-compression/blob/research/gic002-capacity-routing-20261016/experiments/mirror_applications/ma-041-mirror-attention-heads/README.md — aligned logical head feasibility, measured real-runtime loss.
- TM001 https://github.com/Unjuno/functional-quotient-compression/blob/research/gic002-capacity-routing-20261016/docs/phase2/TM001_PARALLEL_PERIOD_TOKEN_MIXING.md — joint within-packet uncertainty boundary.

**ERROR CHECK:** No trained-model result is claimed. No full-T×K mask future leak. Byte saving is not asserted from token flops alone. No inherent gain from symmetric conjugation or from using identical direction codes. Strong native controls are explicit; registry statuses and main stay unchanged.
