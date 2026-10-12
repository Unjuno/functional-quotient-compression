# MA-701-LP01 Rev.2 — 学習可能なMirror変換方向の共同最適化計画

**科学的状態：PLAN_ONLY / 未実行。** 2026-10-12 JST。既存 MA-701 に接続する限定サブ実験。新規MA番号は発行しない。main、公式MA registry、既存FAIL/PROMISING判定は変更しない。**本書の数値は実験計画上の設定・事前判定条件であり、測定結果ではない。**

> Worker instructions: Read the repository EXPERIMENT_CONTRACT.md and MIRROR_PARAMETER_INTEGRATION_DOCTRINE.md first. Implement tests before implementation, confirm a failing test before each feature, run full scoped tests, freeze source/protocol hashes before evaluating each reserved development/fresh world. Do all available computation in the container. Do not claim replication, novel Mirror capacity, or native baseline superiority from an aligned synthetic teacher alone.

## 0. 根拠・既存研究と本計画の差分

- **MV-JOINT-001（2026-10-11）:** 共有2D出力と複数タスクの共同損失・学習角度は実験済み。ただし回転面は固定。相補4タスク BCE 0.08124、非相補4タスクでは固定4方向 BCE 0.31772・独立4ヘッド 0.06562。角度を離すだけでは品質が下がる。
- **TCSMA-001/002 / MA-1194 isolated staging（2026-10-12）:** D、疎係数 A、固定面の角度 m、専用Eをタスクデータから同時学習。部分一致補助損失は固定リッジ推定に主として作用し、5新規seedで0勝（nMSE 0.03236 vs 補助なし0.02638）。新しい学習可能軸・機能勾配・一対一部分対応は未確認。
- **MT001/RF1、MN008、MS009–014、SRM001:** 多View同時学習・再利用・共有/専用の有望な条件は既に検証済み。一方で、学習データを重複提示した効果、非共有対照との最終品質、推論コストの問題が残る。
- **MA-701（P0 UNTESTED）:** 反射ベクトルを学習するHouseholder系列を既に登録。**LP01はその実験計画と関連する学習可能な「共有2D回転面」サブスクリーンであり、MA-701のHouseholder本体を置き換えない。** 学習できるHouseholder反射自体はHRA（Yuan et al., NeurIPS 2024）の既存技術。MA-703（層×タスク反射）・MA-956（学習した対称変換のタスク座標化）とも連携。
- **MA-574/575:** 個別実験が存在。int4量子化用の離散回転コードと選択の実験であり本研究の共同学習とは異なる。574は凍結ソース不整合でNOT_ESTABLISHED、575はnative BOFT等価でFAIL。
- **一番安い既存対照は重要:** HRA、OFT/BOFT、Givens構成、自由係数の共有rank-2/4、FiLM/IA3、同等予算LoRA、単一共有trunk＋独立ヘッド。数式的に同じnative変換をMirrorの新機構とは呼ばない。

参考文献: HRA https://arxiv.org/abs/2405.17484 ; https://papers.neurips.cc/paper_files/paper/2024/hash/cdd0640218a27e9e2c0e52e324e25db0-Abstract-Conference.html

## 1. H — 分離して検証する仮説

**H1 軸の学習効果:** 共有された学習可能な回転面ベクトルが、固定回転面＋同じタスクコードに比べて未見タスク例の損失を下げる。新しい軸を前提に教師が生成される「構造に整列した」合成世界と、オフファミリーを分ける。

**H2 共同表現形成:** 共有FFN、変換面、タスクコードを訓練時から共同更新すると、機能的に異なる複数のラベルを表現でき、単純な角度分散の強制より予測品質と干渉を改善できる。

**H3 物理的な損益:** K=4/8 は表現学習機構の検証、K=16/32 は共有軸の初期費用の償却と完全な実推論ファイル容量を評価。圧縮「成功」は強いnative対照と品質が同等のときに限る。

**H4 スパース・整合拡張:** H1が支持されたときだけ、生成子の疎選択、専用rank-1補正、機能空間の一対一部分対応を一つずつ追加する。以前のTCSMAの補助勾配断絶を再発させない。

**C0 反証:** 生成子を学習しても通常の低ランク/FiLM/HRAで同一の機能をより安く実現できる、あるいは学習済み軸が単なるゲージ自由度ならMirror特有の優位性はない。

## 2. 対象モデルと数式・次元

初期の非線形タスクモデルは、共通trunkが出す h_theta(x) ∈ R^(d_in)、共通 W_up∈R^(d×d_in)、W_down∈R^(d×d) を持つ。変換が作用する隠れ幅 d=32、合成入力 d_in=24、digitsのtrunk出力 d_in=32。出力幅 o は合成タスク=6、digitsの二値判定=1。共有 W_head∈R^(o×d) を用いる。各生成子 j=1..p は、すべてのタスクで共有する生ベクトル a_j,b_j∈R^d から定義する。

1. u_j = a_j / ||a_j||_2
2. vtilde_j = b_j − (u_j^T b_j) u_j
3. v_j = vtilde_j / ||vtilde_j||_2
4. G_j = v_j u_j^T − u_j v_j^T
5. R_j(phi) = I + (cos(phi)−1)(u_j u_j^T + v_j v_j^T) + sin(phi) G_j
6. Q_t = R_p(m_tp) ... R_2(m_t2) R_1(m_t1)
7. F_t(x) = W_head [W_down GELU(Q_t W_up h_theta(x)) + E_t GELU(W_up h_theta(x))] + b_t

**必須の数値訂正:** 正規化分母に「||a|| + epsilon」を使うと厳密な単位ベクトルにならない。||a||、||vtilde||がepsilon未満のときは数値的退化として検知し、そのseedの許可された事前定義した再初期化規則または FAIL を記録する。有効な大きさのときは真のノルムで割る。Gram–Schmidtが成立していれば R_j は理論上直交で、積 Q_t も直交。FP32の許容誤差は検査する。

**必須の勾配訂正:** 全m_t=0で開始すると、生成子軸 a,b への一次勾配は消失し得る。タスクコードは小さい非ゼロ値から初期化し、初期の軸勾配が実測でゼロでないことをチェックする。0初期化は失敗機構の対照としてのみ使う。

**変数・単位:** d=変換隠れ幅32、d_in=共通特徴入力幅24または32、o=タスク出力幅（合成6/digits1）、p=共有変換面数4、K=論理タスク数4/8/16/32。a,b,u,v,h,m∈実ベクトル（正規化NNにおいて無次元）、G/Q/W/E∈行列（無次元）、b_t∈R^o（無次元logit）、合成 L_nMSE は無次元、二値L_BCEはnat/label、B_totalはbyte、推論時間はms。E_t∈R^(d×d) は任意追加rank1の P_t S_t^T（各d×1）で、主検証ではE_t=0。W_head∈R^(o×d) は**全タスク共通**。Q_t(d×d)・W_up(d×d_in)・h(d_in×1)で中間d×1、W_headで最終o×1。変換ベクトル数は共有側2pd=256個、タスクコードKp個、ほかの共有重み・b_t・private・メタデータは全て別途計上する。

**重要な容量損益計算:** 32次元×p=4では軸256個＋K×4個。**scalar出力(o=1)のdigits系**で、最小の独立32次元タスクヘッド K×32 個と「軸+コード」だけを比較しても、K=8では288対256で軸方式が既に大きい。理想化した単純な損益分岐は 256+4K < 32K、すなわち K>9.142... で K>=10。共有readout・private・NPZメタデータまで含む実損益分岐は別に測る。このため **K=8に5%全体容量削減を強制しない**。K=16/32の方が適切な容量検証域となる。

## 3. T — 実験系列・タスクセット

### Phase A：学習機構と最適化を切り分ける（最優先）

**A1 / 数値正当性:** p=1,2,4の回転実装と作用点を unit/gradient/gauge テスト。全ゼロ角で軸勾配が小さくなることを「予想される退化」対照として確認。m≠0では軸、コード、共有Wにタスク損失から勾配が届く。

**A2 / 三つの合成世界（入力d_in=24・隠れd=32・各タスク出力o=6）:** (aligned) 原点不明のランダムな低次元共有回転面を持つ教師、(partial) 6出力行のうち1つだけ共有Viewから外れる教師、(unrelated) 共有不能な独立密教師。各12タスク×128 train例×512 test例、教師軸・潜在パラメータは学習器に与えない。ランダムな入力/教師seedを全分割で独立させる。nMSEおよび最悪タスクnMSEを記録。

**A3 / 非線形の実データ:** sklearn digits の元画像IDを先にtrain/dev/testに分割し、同一画像から8つの教師あり二値判定を作る（偶奇、5以上、素数、3の倍数、0/1/2、3/4/5、6/7/8、1/4/9）。ここは実画像だが、**独立の自然タスク8件ではない**。非相補を主評価にし、相補ラベルだけの4Viewは別の機構対照。評価は平均BCE、最悪BCE、正解率、Viewの機能多様性。同一trunkが1回だけ重いforwardを行うことも確認する。

### Phase B：構造と予算（Phase A の開発判定後に別プロトコルでのみ実施）

固定/学習可能な回転面×p=2/4、軸共有・タスク個別軸、選択top-2・全生成子、rank-0/1 privateをアブレーションする。学習可能軸の追加shared bytesを全て支払う。K=4/8/16/32の合成タスク銀行で完全ファイル容量と同品質条件の損益分岐を測る。プライベート残差率・未使用View・単一viewへのcollapseを報告。

### Phase C：整合損失（Phase A+Bから分離）

以前のTCSMAで失敗した capped 1-to-many 最近傍損失はそのまま再使用しない。訓練入力のみを使う **一対一・未対応ダミー付き部分マッチング**を比較、重み/ベクトル距離だけでなく出力logit/活性のタスク機能差で対応を構成する。固定するのは対応選択の離散マスクのみで、生成された特徴と変換軸を計算グラフから切り離さない。

Cの初回は λ_align=0 を主要対照とし、λ_align>0 の勾配が shared backbone、共有軸、タスクコードへ有限非ゼロで到達するかを機械的に検査。対応損失・Top-2疎化・private補正の寄与を必ず別々に帰属させる。選択された対応のprecision、未対応率、誤対応による最悪タスク損失も報告。

## 4. 必須対照と公平性

| ID | 対照 | 役割 |
|---|---|---|
| C0 | Mirrorなし・共有FFN | タスク依存自由度の必要性 |
| C1 | 固定座標Givens + 学習m | **H1の直接対照** |
| C2 | 学習共有Givens面 + m、整合/privateなし | **主要候補** |
| C3 | native HRA / learnable Householder、共有vector bank | 既存技術による学習ベクトル変換 |
| C4 | native learnable Givensと同じ演算 | 数学的な完全alias監査（同一ならM0） |
| C5 | rank-matched shared lowrank code / coefficient bank | 小さい一般的な自由度との比較 |
| C6 | FiLM / IA3 + 構造非拘束コード | cheap gating |
| C7 | 独立 LoRA rank1/rank2/rank4、source-aware公平版 | 強い既存学習・容量対照 |
| C8 | 共有trunk + 各タスク通常head | **1回trunk計算の公平なnative多出力** |
| C9 | 独立task-local FFN、別学習 | 高容量な品質upper bound |
| C10 | 学習共有Givens + sparse top2/private1（別系列） | スパース/専用残差の追加価値 |

重要：C2とnative学習Givens（C4）が同一関数・同一保存量になることはあり得る。その場合は明確に **native alias / Mirror固有性なし**とする。C2が固定面C1に勝つことは「学習軸が役立つ」機構結果ではあるが、既存HRAに対する新規性の証拠ではない。

**統制:** 同じランダムなshared初期値、同じ元画像/例ID、同じタスクのsampling stream、同じoptimizer update数、比較可能なtrainable-state容量、同じ数のforward/タスク/学習例を提供。各nativeも同じ校正用探索予算を受ける。例数を増やす multi-view と single-view を比較するときは総教師損失評価回数も揃える。fixed-update・equal-active-MACs・near-convergence(長時間更新)の結果を混同しない。

## 5. 事前固定ハイパーパラメータ選択

**主機構:** p=4、private=0、sparse=off、align=0。全て共同学習（軸はshared、m_tはtask別）。

**校正探索のみ:** 軸専用LR ∈ {3e-4,1e-3}、軸を解凍するstep ∈ {0,100}、初期非ゼロ角度の標準偏差 ∈ {0.03,0.12} rad、合計2×2×2=8設定。共有W LR=0.001、AdamW weight decay 1e-4、dropout0。native C1/C3/C5/C6/C7にも同等の校正予算を付与。warmup=100のときもコード/共通重みは学習し、軸のみ最初の100 stepを凍結する。全0角度の単体診断は研究主結果に混ぜない。

**校正:** 合成 seed=17100、および同時にロックしたdigits校正パーティションを使う。最良を選んだら「1つの選択設定」とそのnative対象設定を計測前に保存。

**開発:** synthetic worlds=17101/17102/17103、digits split=同seedで画像ID分離。各方式800 AdamW updates、batch64（合成）/128（digits）。H1候補と最良nativeは同じ初期条件から2400-step継続も登録し、800-step優位とnear-convergence優位を別々に記載。

**Fresh:** 17111–17115（未開封、開発合格時にのみ使用）。実装・seed・teacher/source/protocol hashes・選択設定・閾値を凍結した別コミットの後に一括評価する。開発が不合格ならfreshを開かない。新しい後続実験は新seedを割当てる。

## 6. D — 事前判定閾値・停止条件

**A / 開発→freshゲート（すべて必須）**
- H1 機構: aligned合成worldの固定軸C1比で、C2のテストnMSEが **2/3 seed以上で相対10%以上改善**、かつ候補の平均 nMSE <=0.05。
- H2 実タスク: 非相補digits8役割のC2平均BCEが、比較できる最良のquality-qualified native対照比で **+0.02 nat/label以内**、最悪役割差 +0.05以内、平均正解率差 −1pp以内。結果を単一データ集合に条件付けて扱う。
- H0 正当性: all unit/replay/orthogonality/gradient/gauge checks pass。候補・nativeで較正予算・実保存量・学習コストを隠さない。
- C4のnative完全aliasはH1機構ゲートの検証を妨げないが、**Mirror固有性を否定**するため、採用判定はできない。

**B / Fresh H1機構:** aligned world C2がC1より相対10%以上改善するものが **4/5 seed以上**、median相対改善10%以上。独立ridge/LoRA上限との差も必ず開示。

**B / Fresh H2タスク品質:** digits平均BCEが最良の同等予算nativeより+0.02以内、最悪役割BCEが+0.05以内、正解率差が−1pp以内。各seedごとにも明示し、全seedの平均だけで相殺しない。

**C / K-scaling容量・実行時間:** K=8は機構比較のみで **5%容量削減を成功条件としない**。K=16はブレークイーブン探索、K=32は完全NPZ inference-bankの保存量が、その世界で品質閾値を満たす最安nativeの **95%以下**。合成K32で候補nMSEがnative比の絶対差+0.005以内、かつ0.05以下という品質閾値も必要。CPU batch1とbatch64のfolded推論中央値はnativeの1.25倍以内。eager・変換構築時間も別途報告。K32の合成教師alignedのみで容量PASSでも「自然機能圧縮成功」へ昇格しない。

**総合状態:** H1成立/H2不成立→限定的機構シグナル。H1とH2合格だがnative同値→有用な既存正規直交adaptation、Mirror独自利得 **NOT ESTABLISHED/M0**。上限・非共有・同容量対照のどれにも品質と容量が勝てなければFAIL。freshを開かなかった世界の判定はdev-only。H3は別途家族の実用的なパレート点で評価する。これらの判定条件を結果後に修正しない。

## 7. C/U — 失敗モード・誤差・監査

- **初期m=0で軸勾配ゼロ**、ほぼ平行なa/b、タスクコードの飽和、p面間の実質同一化、View全滅/崩壊、shared後段の表現不足をそれぞれ監視。
- **ゲージ偽装:** 同じ回転面の異なる基底(u,v)は同一作用を表せる。生の軸cosineだけで学習成功を判定しない。projector u u^T+v v^T、同一入力logit差、Jacobian dF/dm、実タスク損失で判定。
- **最適化予算:** 候補だけに追加更新や長いLR探索を与えない。失敗をパラメータ化/学習性/容量のいずれと解釈できるか分離する。
- **記述量:** NPZには学習済みtrunk、共有生成子、生軸、コード、利用条件、private、メタデータ、decoderをすべて格納。再構成可能な固定定数は宣言したアルゴリズムとseedを記録。training stateは推論容量に混ぜないがoffline費用を別途記録。
- **時間:** Torch単一スレッドFP32、CPU仕様/clock/cgroupを記録。サンプル数・batch=1,64・ウォームアップ/繰り返し・時刻/分散を含め、マイクロベンチだけからGPUの速度を主張しない。
- **統計:** 同じdigitsデータセットの分割seedは母集団独立研究ではない。paired task-wise差、seed別値、中央値・範囲、およびbootstrap（必要なら記述的と明記）を提示。npzの数値リプレイとtraining再実行は別の証拠。

## 8. 実行可能な作業単位と具体的ファイル

Project dir: experiments/mirror_applications/research_intake/ma701_learnable_planes_20261012/

### Task 1 — プロトコル/データファイアウォール
Files: PROTOCOL.json; PRE_RUN_LOCK.json; source/worlds.py; tests/test_worlds.py.
- [ ] Test first: synthetic train/test RNG stream disjoint, digit image IDs disjoint, target teacher axis hidden from learner, no fresh IDs accessed.
- [ ] Observe RED with pytest; implement make_synthetic_world(seed:int,family:str,cfg:dict)->dict and make_digit_world(seed:int,cfg:dict)->dict, observe GREEN; commit.

### Task 2 — 変換面の学習器
Files: source/planes.py; tests/test_planes.py.
- [ ] Test first: Q(0)=I, Q^TQ≈I within FP32 1e-5, preserves norm, reversal works; degeneracy detection; multi-plane order noncommutativity, finite-difference/autograd correspondence in float64.
- [ ] Explicitly test zero-m expected vanishing axis gradient; nonzero-m task gradient flows to raw a,b and m; observe RED then GREEN.
- [ ] Implement orthonormal_planes(raw_a,raw_b,eps:float) and apply_plane_views(x,raw_a,raw_b,angles), using O(p*d) updates per vector without d×d materialization in the operational path; dense reference only for tests; commit.

### Task 3 — 比較可能なタスクモデル
Files: source/models.py; source/train.py; tests/test_train.py.
- [ ] Test first: 1 physical shared trunk, all K task outputs, all trainable states counted; gradients into shared FFN/axes/codes, and optional private code only when present.
- [ ] Implement make_method(method:str,cfg:dict)->nn.Module and fit_world(model,world,cfg,seed)->dict with same sampling/order/training budget; implement C0–C9 controls and 800/2400 train logs; commit.

### Task 4 — 保存形式/計測
Files: source/codec.py; source/evaluate.py; tests/test_codec.py.
- [ ] Test first: deterministic NPZ state can be reconstructed after reload with identical model outputs and full byte length/hash, all code/generator/private tensors present.
- [ ] Implement full inference-bank serializer, paired quality metrics, eager vs folded benchmark batch1/64, seed logs, K-scaling. Run tests; commit.

### Task 5 — 調整と開発判定
Files: RESULTS_CALIBRATION.csv; PROTOCOL_SELECTED.json; RESULTS_DEV.csv; VERIFICATION_DEV.json.
- [ ] Execute only calibration seeds from fixed 8-setting grid; select candidate/native params without dev/fresh peeking. Save selection and source/data hashes before dev.
- [ ] Execute all three dev seeds and baseline methods, preserve every negative result. Decide according to Section 6. If FAIL, stop fresh but report design diagnosis; do not promote by changing thresholds.

### Task 6 — 別途の疎化・整合アブレーション
Files: source/alignment.py; tests/test_alignment.py; PROTOCOL_ABLATION.json.
- [ ] Only as a separately locked experiment; compare λ_align=0 with nonzero, one-to-one partial matching/unmatched dummy, sparse top-2, rank1 private independently; confirm its own gradient routes.
- [ ] Report assignment cost, false alignment, per-task degradation, serialized private fraction; never use the 17111–17115 fresh seeds for this tuning.

### Task 7 — Fresh検証/台帳提出
Files: PROTOCOL_FRESH_LOCK.json; RESULTS_CORE.csv; STATUS.md; VERIFICATION.json; README.md.
- [ ] If the dev gate passes, lock frozen source/selected hyperparameters; run all reserved fresh seeds without adapting or checking thresholds early. Save numeric payload/replay for every model.
- [ ] Rerun at least one full training seed bit-for-bit where deterministic; run all scoped and available historical repo tests. Report failures instead of skipping.
- [ ] Commit source/protocol/tests/results/hash-checked evidence to separate research branch; update existing MA-701 status only when the full MA experiment contract is met. Do not alter main. Report H/T/D/C/U and exact applicability bounds.

## 9. 完了条件・次の判断

This **planning revision** is complete only after the GitHub file and RUN_MATRIX.csv can be read back and verified; this does not mean an experiment passed.

Begin implementation with Tasks 1–4, then run calibration before new development seeds. **Do not repeat MV-JOINT, fixed-angle TCSMA or discrete SpinQuant selection as the new contribution.** New question: Do shared learnable plane vectors, jointly optimized with task loss, reduce actual task loss/byte/time versus both fixed planes and the strongest native operator families, without hiding a task-specific full weight copy?

**実験は未開始。** The next action requires implementing and preregistering code/protocol, not opening the reserved seeds.
