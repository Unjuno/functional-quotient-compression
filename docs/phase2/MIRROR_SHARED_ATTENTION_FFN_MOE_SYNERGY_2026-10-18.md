# Mirror共有Attention × FFN × MoE × 反復：統合研究仮説（2026-10-18）

**状態：研究目的と検証計画（DRAFT、結果ではない）。** 既存の数理検証 MA-691、MA-1162 と、FFN/演算融合実験 MA-1163/1164 を統合して評価する。履歴のPASS/FAILや凍結プロトコル、mainは変更しない。

> **Mirror insertion:** shared physical Attention/FFN cores are accessed through compact role-, head-, expert- and depth-specific Mirror coordinates interleaved with nonlinear activations, aiming to replace replicated logical operators without storing a full independent tensor for every role. Measure all View/router/private/code bytes and compute.

## 1. 研究目的

**最大の論点は「重みを共有する」だけでなく、共通の入力・中間結果・正準KVキャッシュも共有して、異なる機能を差分計算で取り出せる範囲を拡張すること。**

構造：
1. 一つ（または少数）の共有物理核 S から、入力Hに対する基底Q/K/VやFFN-up等を計算する。
2. 同じHからK個のMirror Viewを並列に評価。Attentionの有効Query/K/V関係、非線形FFNの活性方向、専門機能を変える。
3. 集約・残差更新後、新しい状態で同じ核をR回反復して深さを実現する。
4. セルフAttention、クロスAttention、Encoder/Decoderの役割は入力元・マスク・履歴の依存関係を明示してから共有対象へ加える。

**「一つの共有パラメータオブジェクト」と「独立学習スカラーが1個」は別。** 本流はm成分の物理ベクトル・共有核を用い、m、数値精度、Mirrorコード、並列数、反復数を独立に制御する。1スカラー多倍長方式は別系列。

## 2. 共有可能性の五つの階層

| 階層 | 何を再利用できるか | 成立条件 | なお必要な費用 |
|---|---|---|---|
| S0: 重み | 基底物理Attention・FFN核 | 全役割が同じ物理学習状態を参照 | 役割コード、変換、私有残差 |
| S1: 射影 | 同一Hからの共通QKV／FFN-up | 同一Hと同一前処理、分岐は射影後 | 各Mirror、ゲート、後処理 |
| S2: KV | K0/V0一組の実キャッシュ | 同じproducer・token履歴・層・反復深さ、既知の合法なView代数とRoPE | View別Queryと出力処理、スコア |
| S3: 注意確率 | softmax後の共通注意分布 | 有効スコアが位置ごとの共通定数差まで一致 | Value処理・出力変換 |
| S4: 注意出力 | 共有 a V0 | S3が成立し、Vの差が後置可能な**線形**右作用 | 各ViewのB作用と非線形出力 |

**同じ重みから複数の機能が生まれる**ことは、S1～S4が成立することを意味しない。異なる注意スコアならK個のsoftmax/重み付き集約が必要。単一のGPU呼出しでKヘッドを処理しても算術演算量がゼロになるわけではない。

### Attentionの厳密な条件

\[
K_j=K_0A_j,\quad V_j=V_0B_j,\quad
o_j=\mathrm{softmax}\!\left(\frac{(q_jA_j^\top)K_0^\top}{\sqrt{d}}\right)V_0 B_j.
\]

単一producerのK0/V0を持つ場合、過去のView別K/Vを毎回物理保存する必要はない（既存MA-691）。ただし、固定A/Bは通常のQuery/出力投影へ代数的に吸収できる場合があるので、**native folded Query/Output、MQA、GQA、MLA**を同品質・同バイト対照にする。これはKV共有の数理成功であってMirror独自の速度勝利を証明しない。

非線形の過去Value変換は一般に注意加重和の外へ移せない。Encoderの双方向、Decoderの因果マスク、Crossの異なるQuery/Key/Value元状態は、それぞれ区別する。Encoder完了前の依存するCross-Attentionは同時には完了できない。

### FFNの共通処理と複数Mirror

\[
z=hW_\mathrm{up},\qquad
f_j(h)=\phi_j(zQ_j)Q_j^{-1}W_\mathrm{down},\qquad
h^{(r+1)}=h^{(r)}+\frac{1}{K}\sum_{j=1}^{K} f_j(h^{(r)}).
\]

この形なら、同一hに対するW_upの射影zは一回でよい。異なるQ/φが意味のある役割差を作れば複数FFNとして機能し得る。ただし各jの非線形変換とW_down適用、rの反復は計算が必要。MoEのrouter/top-k/private residualとメタデータは別途課金する。固定Qを普通の線形射影へ吸収すること、等価な非Mirrorゲート/低ランク基底で再現できる可能性も必ず対照に残す。

### 変数表（数式の前提）

| 記号 | 意味（日本語） | SI単位 | 定義域・形状・前提 | 型 |
|---|---|---|---|---|
| \(S\) | 学習・保存する共有物理核 | 1 | m個の独立係数等を含む | テンソル |
| \(m\) | 共有核の独立係数数 | 1 | 正整数 | 整数 |
| \(h,h^{(r)}\) | 入力・第r反復の隠れ状態 | 1 | 1×d | 行ベクトル |
| \(z\) | FFN共有up射影結果 | 1 | 1×f | 行ベクトル |
| \(W_\mathrm{up},W_\mathrm{down}\) | FFN共有基底 | 1 | d×f、f×d（行ベクトル規約） | 行列 |
| \(Q_j\) | j番FFN Viewの可逆Mirror | 1 | f×f | 行列 |
| \(\phi_j\) | j番の非線形活性 | 1 | \(\mathbb R^f\to\mathbb R^f\) | 写像 |
| \(f_j\) | j番の論理FFN | 1 | d次元の入出力 | 写像 |
| \(K_0,V_0\) | 共通の履歴Key/Value | 1 | T×d、同producer | 行列 |
| \(K_j,V_j\) | j番の論理K/V | 1 | T×d | 行列 |
| \(A_j,B_j\) | K/Vの既知の右作用 | 1 | d×d、固定線形（必要時RoPE整合） | 行列 |
| \(q_j,o_j\) | j番のQueryとAttention出力 | 1 | 1×d | 行ベクトル |
| \(d,f,T\) | 状態幅、FFN幅、履歴長 | 1 | 正整数 | 整数 |
| \(K,R,j,r\) | 論理View数、反復数、各番号 | 1 | 正整数／範囲内添字 | 整数 |

**形状・単位チェック：** \(h(1\times d)W_\mathrm{up}(d\times f)=z(1\times f)\)。\(z Q_j\to\phi_j\to Q_j^{-1}\to W_\mathrm{down}(f\times d)\)で1×dとなり残差へ加算できる。Attentionは(1×d)(d×T)→1×T、(1×T)(T×d)→1×d。内部特徴は無次元に正規化した理論モデル。

（行ベクトル規約で\`f_j(h)=phi(z Q_j) Q_j^-1 W_down\` と読む。）

## 3. 優位性を反証可能に調べる

| 順序 | 必須試験 | 比較方式 | 何が得られれば進むか |
|---|---|---|---|
| P0 数理 | S0–S4の部分共有、attention head divergence、同Q/K gauge、RoPE、FFN非線形の差 | materialized/absorbed nativeとMirror | 機能等価な箇所を誤差で証明し、異なる箇所は区別 |
| P1 最小学習 | 一核＋役割コードで2/4/8個のFFNとヘッドを学習、物理m・幅・コード数を分離 | independent/tied、FiLM、LoRA、gain、MoE | 適合教師だけでなく非適合課題で品質/byteのPareto改善 |
| P2 反復 | R=1/2/4（候補）、K別の機能維持・干渉・停止時出力 | Universal Transformer、反復のみ、私有残差 | 反復の実品質利得が演算費を上回る |
| P3 KV | 同一履歴K/Vを物理aliasしたend-to-end prefill/decode、役割切替、前段Viewでの無効化 | GQA/MQA/MLA、native folded、materialized reference | 実メモリ/実tok/s改善と同NLL |
| P4 統合 | Attention+FFN+MoE、最後にCross/Encoder/Decoder、学習Router | 最適化済みnative、強いMoE、適切なprivate | 強い対照に対する同品質の総量/速度利得 |

**優先:** 現行[作業キュー](MIRROR_SHARED_CORE_EXECUTION_QUEUE_2026-10-18.md)のQ0/Q1で既存multiheadの凍結研究を監査・完了 → Q2のFFN共通計算を検証 → Q3のKV → Q4で組合せ → Q5 Cross。すべてを一度に実装しない。

- H: 共有物理核からの有用な機能差と、共通射影/正準状態の再利用により、品質が同等の従来手法より高い性能・保存効率を実現できる可能性。
- T: まず機構と予算の監査、次に3独立評価世界×2 seed以上の比較。開発データでのみ選択し、評価前にプロトコル固定。
- D: 同保存量・同演算予算のnative/FiLM/低ランク/共有反復対照へのPareto改善が必要。**同じ結果を得るための単なる通常foldingだけならMirror固有効果としてFAIL**。
- C: 速度差は重み・KV共有やfoldingによるものでMirrorなしで達成できる、あるいは共有できないprivate機能の損失が大きい。
- U: 継続学習安定性、View間干渉、実KV producer/深さの同一性、Crossの参照情報、実GPUカーネル、プロジェクト固有のデータ・モデル。校正済み複合不確かさu_cと包含係数kは未推定。

## 4. 既存の根拠と限界

- **MA-691:** 互換な線形右作用の正準KV読み出し等価性PASS。任意のEncoder/Decoder中間状態を共有できる証明ではない。
- **MA-253:** 早い層のMirror変化が下流KVを変更する反例。全層で無条件共有は不可。
- **MA-1162:** multihead共通QKVの数学的機構と学習用合成課題を検討中。異なるAttention機能が完全に1回の演算だけで得られることは未実証。
- **MA-1163/1164:** Mirror仮想MLP変換とCPU融合の技術的成立。通常MLPへの同予算・同品質のMirror固有優位は未確認。
- **一次資料:** Vaswani et al. arXiv:1706.03762; Dehghani et al. arXiv:1807.03819; Ainslie et al. GQA DOI:10.18653/v1/2023.emnlp-main.298。

**この文書を新しいPASS結果や既存MAステータス変更として扱わない。** 仮説、コード、反例の境界と新規測定を将来分離して保存する。
