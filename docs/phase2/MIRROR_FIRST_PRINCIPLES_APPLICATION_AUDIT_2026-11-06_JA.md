# Vector Mirrorの第一原理・適用範囲・実証ロードマップ：リポジトリ横断監査

**監査日: 2026-11-06 / 位置付け: 複数研究ブランチの記録に基づく証拠整理。**
本書は新規学習実験ではなく、過去のPASS/FAIL、Fresh/OOD、推論実バイト、CPU計測を監査した研究統合である。研究台帳の科学的ステータスを変更しない。mainへマージしない。一次データは各実験の凍結プロトコルと検証記録を優先する。

## 0. 結論（何に使えそうか）

**第一候補は「既存の状態・演算をViewごとに複製せず、低記述長の読み出し変換で再利用する」用途。**
とくにKVキャッシュのcanonical表現とlazy readは、条件を満たす場合に代数的に正しい。第2候補は共有入力/出力因子を持つアダプタ群のコード統合。第3候補は、自然に学習された関数間の幾何学的整列性を事前検査したうえで行うexpert/MLP/Private残差の共有。

「Mirrorが任意の別機能を詰め込める」「再帰段数を増やすだけでMirrorがLoRAの代用になる」「有望判定54件が実モデルで勝った」は**支持されない**。

## 1. リポジトリ調査の範囲と信頼度

- 調査基準：研究ブランチ research/loop-lora-mirror-20261105 および各専用実験ブランチのCLAIM_LEDGER・STATUS_BOARD・一次README・検証記録。2026-11-06の新統合ブランチは research/mirror-application-audit-20261106。
- 中央台帳：1,184候補、54 PROMISING、149 FAIL、4 SCREENING、977 UNTESTED。54 PROMISINGの多くは**初めからView構造に整合する合成教師**を評価する条件付き機構証拠。自然言語/GPU/一般圧縮の採用証拠とは別。
- main の最終確認先頭コミットは2026-10-06の0c3e5d78。実験結果の最新版は研究ブランチ群。mainを古い研究状態だと断定せず、**mainと各研究ブランチの時系列・役割が異なる**と認識する。
- 同じ日付のLoop比較は**異なる2系列**。160モデルと185モデルを同じプロトコルや独立追試としてまとめない。後者には新たな5件の同予算FiLM+rank1事後比較がある。

### 証拠の水準

| 水準 | 意味 | この研究での例 |
|---|---|---|
| **A: 代数的機構** | 明示条件の下ですべての対象入力に恒等式が成立、数値検算あり | MA-691のcanonical KV; Householderの直交性と階数制約 |
| **B: 整合した合成世界の成立** | 同じ構造の教師に対するFresh結果・保存量・再生確認 | MA-241、244、245、189、012など |
| **C: 非整合/独立関数の学習診断** | より中立な教師で、対照と限定された予測能力を直接比較 | MA-1183、MA-1184開発、Loop-LoRA 2系列 |
| **D: 実運用Pareto実証** | 自然なタスク、強い既存方式、推論質と実時間・GPU/メモリまで優位 | **現時点では未確認** |

BはAの代わりではなく、CもDを保証しない。合成整合だけの成功は自然分布に外挿しない。

## 2. 物理機構の第一原理：何ができて、何ができないか

以下は正規化された特徴/重みを扱いSI単位はすべて1。位置やトークン数、計数も無次元。

| 記号 | 日本語での意味／定義 | SI単位 | 定義域・前提 | 型 |
|---|---|---|---|---|
| \(h\) | 共有核へ入る隠れ状態 | 1 | 実d次元 | ベクトル |
| \(d,r\) | 隠れ次元、反射数 | 1 | 正整数、\(r\le d\)を通常想定 | スカラー |
| \(u_i\) | 第i反射の単位法線 | 1 | \(\mathbb{R}^{d}, \|u_i\|_2=1\) | ベクトル |
| \(H_i,Q_m,I_d\) | Householder反射、m番目の合成Mirror、単位行列 | 1 | d×d、直交Q、\(H_i=I_d-2u_iu_i^\top\) | 行列 |
| \(m\) | 論理View/機能ID | 1 | 離散整数 | スカラー |
| \(\sigma\) | 非線形活性化 | 1 | ReLU/SiLU/tanh等 | 写像 |
| \(\Phi_m\) | View別の非線形応答 | 1 | \( \mathbb{R}^d \to \mathbb{R}^d\) | 写像 |
| \(T,d_k,d_v\) | 接頭系列長、key幅、value幅 | 1 | 正整数 | スカラー |
| \(q,K,V\) | 一つのQuery、canonical Key/Value cache | 1 | \(q\in\mathbb R^{1\times d_k}, K\in\mathbb R^{T\times d_k}, V\in\mathbb R^{T\times d_v}\) | 行列（qは行ベクトル） |
| \(A_m,B_m\) | Key/Valueの論理View右変換 | 1 | \(d_k\times d_k\), \(d_v\times d_v\) | 行列 |
| \(p_m\) | View mのAttention確率 | 1 | \(\mathbb R^{1\times T}\) | 行ベクトル |
| \(F\) | 共通の線形演算核 | 1 | 適合する実行列 | 行列 |

### 2.1 非線形境界の再配置

\[
\Phi_m(h)=Q_m^\top \sigma(Q_mh).
\]

ReLUの場合は、\(Q_m^\top\mathbb R_{\ge0}^{d}\) という回転した非負直交錐へのEuclidean射影になる。**Qが情報を捨てるのではなく、Qで移した座標でReLUが成分を選択する**。線形活性化のみでは前後のQが相殺する。

合成反射 \(Q_m=H_r \cdots H_1\) では

\[
\operatorname{rank}(Q_m-I_d)\le r .
\]

**証明:** すべての\(u_i\)に直交するwでは各\(H_iw=w\)、したがって\(Q_mw=w\)。これらwの部分空間の次元は少なくとも\(d-r\)。従って\(\ker(Q_m-I_d)\)の次元は少なくとも\(d-r\)、階数は高々r。□

つまり、二反射はたとえば幅48で高々二次元の可動部分空間を持つ。非線形性による出力差は広がり得るが、**任意の48次元回転・任意の機能変更を許すわけではない**。必要な差が拡大縮小・独立バイアス・別アルゴリズムならFiLM/LoRA/privateが自然に強くなり得る。

単一Householderは、係数\(\beta=-2\)の結合rank1変換 \(I_d+\beta u u^\top\) と厳密に一致する。Mirror固有の表現力上位性はない。帰納バイアス、実バイト、学習量、実速度でのみ選好を正当化できる。

### 2.2 KVキャッシュ共有：厳密に「計算の場所」を移せる用途

View側でKeyとValueを事前に materialize した場合、
\[
K_m=K A_m,\qquad V_m=V B_m.
\]

同じsoftmax温度・マスクで、

\[
\begin{aligned}
p_mV_m
&=\operatorname{softmax}\left(q(K A_m)^\top/\sqrt{d_k}\right)VB_m\\
&=\left[\operatorname{softmax}\left((qA_m^\top)K^\top/\sqrt{d_k}\right)V\right]B_m .
\end{aligned}
\]

**証明:** 転置の積順序と行列積の結合則により\(q(KA_m)^\top=(qA_m^\top)K^\top\)、また\(p_m(VB_m)=(p_mV)B_m\)。softmaxに渡す各スコアが同一なので確率は同一。□

**重要:** この等価性は、A/Bが直交でなくても成立する。「Mirrorを使えば性能改善」ではなく、**既知のView変換を読み出し側へ移して、過去のKey/ValueをViewごとに保持しない**ための恒等式。先行FFNやAttentionが過去の隠れ状態自体を別物にしているときには、そのまま適用できない。RoPEや正規化、cacheの量子化が絡む場合は変換の順序・可換性を個別に検査する。

**単位・次元検査:** \(qA_m^\top\)は1×\(d_k\)、\(K^\top\)は\(d_k\)×T、softmax結果は1×T、VはT×\(d_v\)、B_mは\(d_v\)×\(d_v\)、最終出力は1×\(d_v\)。いずれも無次元で合致する。

### 2.3 どうして一回の重い計算から多数の論理出力があり得るか

共有入力射影と複数の軽量Viewを分けた場合、複数Viewが**同じ入力を使う**とき、共通の線形射影 \(Ah\) は一度だけ計算できる。ただし各Viewの非線形活性化・出力射影・追加変換は費用を持つ。「複数機能を一本のforwardと同じ費用で同時実行」は成立しない。

学習済み別関数を無条件に同一核へ移せるわけでもない。自然関数群が適合するかの診断、ならびに最強の非Mirror対照（FiLM、coupled rank1、独立小型核）が必要。

## 3. 読み直した実験証拠（選択バイアスを明記）

| 実験/用途候補 | 確認済み数値・事実 | 有効と考える条件・未立証範囲 |
|---|---|---|
| **MA-691 canonical KV lazy read** | FP32最大差3.13e-7、3/3 fresh。T=8192でcanonical K+Vは33,554,432B、8独立View cacheなら約268.4MB、canonical+8 code約33.57MB。CPU eager lazy attention 1.796ms、materialized 1.639ms、materialize変換42.85ms。break-even約342 tokens | **A** 既知右変換。View切替が多い・cache長い・複数Viewを保持したいとき。GPU、実LM、自然に学習された専門家がこの関係を満たすかは未確認 |
| **MA-608 adapter code fusion** | 4096例の線形screenで誤差最大2.4e-7。5,529B vs independent 15,645B、generic matrix code 6,233B。CPU直接融合1.23–1.27ms vs shared AdapterFusion4.06–4.17ms | **A/B** 入出力低ランク因子が共有でき、線形合成が前処理可能な場合。非線形・学習済みAdapterFusionでは未立証 |
| **MA-244/245 K/V role/layer views** | 意図的整合教師でcache bytesが対照の半分。モデル本体保存量の厳格な削減ゲートには未達、CPU速度は悪化 | **B** 役割やレイヤが共通の小変換から導ける。自然の独立したhead/layerを置換できるか不明 |
| **MA-241 tied experts + Views** | 整合教師で24,286B vs untied 45,681B（46.8%減）、3/3品質ゲート。CPU推論はhard tying比0.61x | **B** 関連Expertの座標系だけが違う場合。自然MoE/疎top-kの採用証拠ではない |
| **MA-189 few-shot View adaptation** | 人工整合タスク64例、684B vs LoRA2 851B。incremental state55B vs228B | **B** 固定核で変換方向のみが必要な限定領域。無関連関数ではprivateが必要、実行速度不利 |
| **MA-012 shared+private expert** | 整合教師で8,347B vs独立18,218B、3/3品質・保存量ゲート。CPU速度不利 | **B** 共通核をViewで扱い、非整合残差を小さなprivateで担当させる候補 |
| **MA-330 KV+private residual** | 3整合層＋1非整合層の合成cacheで36,458B vs独立131,752B。ただしgeneric cos/sin対照36,488Bと同品質 | **B** off-orbit変化をprivateで残すことに価値。Mirror固有の差はわずか、GPU速度未実証 |
| **MA-1183 nonlinear function reuse** | 設計時整合教師でReLU Mirror/FiLM比0.6583、しかし部分独立γ=.35主比較0/5 FAIL。別140モデルのγ掃引はposthoc | **C** 整合度が低下すると利益消失。γ=.2〜.3付近の反転は普遍的境界ではない |
| **MA-1184 L0/L1共同学習** | 非Mirror教師の開発3課題×5世界で共同学習Mirrorは固定核・コード片側学習を5/5改善。Mirror固有の対照一括勝利は各0/5、FiLMがしばしば低誤差/高速 | **C** 共有核とコードの協調は重要。ただしMirror固有ではない。L2新規課題族の確認は未実施 |
| **Loop-LoRA-Mirror 別系列160モデル（120共同+40置換）** | R8同種グラフMirror+rank1正答率中央値94.73%、主R4 Mirror固有0/5 FAIL。LoRA2途中FFNをrank1で92.93%誤差回復、Mirrorは28.14%。R8構造OOD全30モデル正答率0% | **C** Mirrorによる一般アルゴリズム獲得・LoRA置換は未立証。Loop/計算費用と構造OODを必ず分ける |
| **Loop-LoRA 別系列185モデル（165訓練+20無学習）** | R8 4-hop rank4教師99.15%、Mirror65.76%、FiLM67.51%、rank1 LoRA93.29%、rank2 LoRA97.53%。Mirror置換・特異優位ゲート各0/5 FAIL。事後同bytes FiLM+rank1との比較でMirror+rank1のCE比0.99027（勝ち2/5） | **C** 真の段別更新には低ランク変更が必要。凍結核の条件であり共同学習の一般否定ではない |

主要一次記録:
- [MA-691](https://github.com/Unjuno/functional-quotient-compression/blob/research/ma-691-lazy-kv-mirror-20261007/experiments/mirror_applications/ma-691-lazy-kv-mirror/README.md)
- [MA-608](https://github.com/Unjuno/functional-quotient-compression/blob/research/ma-608-adapterfusion-direct-mirror-compose-20261009/experiments/mirror_applications/ma-608-adapterfusion-direct-mirror-compose/README.md)
- [MA-241](https://github.com/Unjuno/functional-quotient-compression/blob/research/ma-241-expert-tying-mirror-20261007/experiments/mirror_applications/ma-241-expert-tying-mirror/README.md)
- [MA-1183](../../experiments/mirror_applications/ma-1183-native-function-sharing/REPORT_JA.md)
- [MA-1184開発結果](../../experiments/mirror_applications/ma-1184-natural-function-orbit-diagnostic/runs/2026-11-01-coadaptation/RESULTS_SUMMARY_JA.md)
- [Loop 160モデル](../../research_intake/loop_lora_mirror_20261105/RESULTS_SUMMARY_JA.md)
- [Loop 185モデル](../../experiments/loop_lora_mirror_20261105/RESULTS_SUMMARY_JA.md)

## 4. 研究から予測できる適用範囲

**Priority 1 — canonical KV / shared cache with lazy View:** 理論が先に成立し、使い方も明確。条件付き注意・複数役割・短周期切替など、Cacheの物理複製が支配的な場面で候補。RoPE/MQA/GQA/MLAと比較し、品質とメモリ・GPU帯域・切替頻度の損益分岐を確認する。

**Priority 2 — 多数の線形adapterのコード統合:** 同じ入力に複数源のAdapterFusionを適用し、共有低ランク因子を持つ場合は前段でコードを合成して1回の実効変換にできる。自然タスクで共有基底の適合度が十分か、非線形ボトルネックで誤差が増えないかが判定条件。

**Priority 3 — related expert/MLPと小Private:** 構造的関連性がある場合のみ候補。学習済みExpertの関数空間距離とGram/特異値を事前検査し、Mirror+privateをFiLM+private/coupled-rank1+privateに同bytes・同FLOPsで比較する。

**Priority 4 — recurrent depthでのView切替:** 反復回数を増やす効果はあるが、LoRAの置換に失敗した。今はR8の数値を増やすより、**学習時から多様なグラフ接続・cycle長・hopを混在**させること、そしてrank1/2 LoRAと同FLOPsで比較することが先。これを飛ばすとshortcutの測定になる。

**非推奨（現時点）:** 無関係なExpert/Layerの無条件一括Mirror化、Mirrorでの任意LoRA置換、無条件NoPE全置換、全Viewを毎回実行しながら「演算一回」と数える性能主張、たまたま整合教師の成功例だけで一般適用を宣言すること。

## 5. 次の実証計画：最初に採用判定へ近い用途を検査

**第1の実験:** MA-691を基礎とする**設計時にcanonical KV Viewsを持つ小型実際の因果Decoder**。これは既存の他モデル間cache翻訳ではなく、元から共有を許すアーキテクチャである。比較: 強いGQA/MQA/MLA、事前にmaterializeしたView、lazy View、hard tying、同bytes scalar/generic projection code。評価: Fresh NLL/accuracy、長さOOD、KV正しさ、物理cache resident bytes、切替回数とprefill/decode wall、GPUを使用可能ならbatch/length別のkernel性能。既知A/B変換で数学的exactnessと、学習で品質が保たれるかは別ゲート。最初はCPU機構・実バイト、その後GPU高速化を独立評価。

**第2の実験:** 学習済みAdapterFusion銀行を入手/小規模学習して、**共有低ランク基底**を可能なだけ実出力空間で近似し、コード統合を行う。生教師をMirrorから生成しない。入力条件の選択・非線形誤差・未使用タスクとbytesを報告。

**第3の実験:** その後、MA-1184の独立学習関数を用いてMirror+Privateを試す。原因を区別できない状態でMoE/Loopを追加しない。

### H / T / D / C / U

**H:** 既知の右変換関係が成立する複数のAttention Viewでは、同一のcanonical KVから論理Viewを再現でき、cacheの物理複製を減らせる。自然に学習した複数の機能へ一般化すると、効果は幾何学的適合度に依存し、FiLM/LoRAに負ける世界もある。

**T:** MA-691の完全一致検算から始める。独立パイロットで非飽和・mask/RoPE/LNの整合・実byteを検査。5開発seed、異なる系列・課題の10確認seedを**確認を始める前に凍結**。推論のcache長・View数・切替頻度を段階的に変える。3体系（元の物理共有／条件コード／一般低ランク）の表現差だけを観測できるよう設計。ベンチマーク時はHW、clock、batch、precision、thread、backendを必ず報告。

**D:** 代数的照合max abs <2e-6（FP32の指定条件）を機構PASS。品質については非Mirror強対照とのFresh NLLの許容劣化と、実resident bytes・GPU decode速度を**開封前に**閾値固定。同品質でもcache/速度のPareto優位が無ければ実用採用しない。LoRA等自然関数で成立するという主張は別ゲート。

**C:** 自然なK/Vが同じ右変換で表せず品質劣化、既存GQA/MLAがParetoで勝つ、View切替が少ないのでcache materializeが速い、融合カーネルに追加データ移動がある、複数の過去状態そのものが異なるため同一cacheから復元できない。

**U:** 精度/量子化、mask・positional transformの非可換性、共有コードの学習不安定、最適化初期値、token/sequence分布、CPU↔GPU実行方式。実測ごとに反復中央値/IQRを報告し、独立seedの対応ログ比の標準誤差を記録。拡張係数k=2を小標本の95%CIと同一視しない。

## 6. ERROR CHECK と次の作業者向け注意

- この統合は新しい訓練ではなく**一次実験記録の比較**。160系と185系は異なる条件であり、同じ研究の反復確認と呼ばない。
- 新しいFP64検算は [verify_invariants.py](mirror_2026_11_06/verify_invariants.py) と [verification.log](mirror_2026_11_06/verification.log) を参照。KV algebra 256、d=32のHouseholder rank≤rをr=1–4で256、rank1への包含128を検査。すべてPASS。自然LM/GPU性能の証拠ではない。
- MA-1184の中央 status はなおUNTESTED（L2未実施）。L0/L1開発記録は別の証拠レーンとしてリンクし、statusを研究者の判断なしに変えない。
- 旧MA-1181/1182の構造的 FAIL と不可能なFresh判定閾値を保持。結果を見た後に閾値を書き換えない。
- 新しいMA番号を追加・予約していない。新規確認実験を立てる時点でlive registryから採番する。
- 目的は「用途を予測して実証」することであり、成功した用途を後から探して定義することではない。
