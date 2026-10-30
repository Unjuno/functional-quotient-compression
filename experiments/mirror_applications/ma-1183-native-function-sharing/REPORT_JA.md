# MA-1183 — 第一原理の共有非線形Mirrorの直接学習検証（2026-10-30）

**主判定：FAIL（5 seed中0）。** 210/210モデルを訓練し、全チェックポイントをFresh/OOD生成前にSHA256封印。210/210 Fresh/OOD値を完全再生。**Mirror整合的世界での限定陽性結果と、部分整合・非整合世界での否定的結果を同時に保持する。** 本報告は小型の解析関数課題であり、自然言語LM、Attention、Loop、MoEを実装した性能実験ではない。

## 1. 研究の基本設計と変数表

**目的:** 物理核を増やさず、条件コードで「同じ基本非線形演算を異なる特徴方向に再利用」できるか。それがFiLM・結合rank1・同バイト級独立MLPより学習上優れるか。

下記の入力・出力・重みは正規化された**無次元量（SI単位1）**。パラメータ数・例数・更新回数も個数であり無次元。\

| 記号 | 日本語の意味・定義 | SI単位 | 定義域・型 |
|---|---|---|---|
| x | 入力の観測特徴 | 1 | R^8ベクトル |
| m | 条件／機能ID | 1 | {0,1,2}離散スカラー |
| K | 同時に学習する論理機能数 | 1 | 3、整数スカラー |
| h | 共有隠れ層の幅 | 1 | 12、整数スカラー |
| A,b | 共有入力層・バイアス | 1 | R^{12×8}行列、R^12ベクトル |
| B,c | 共有出力層・バイアス | 1 | R^{3×12}行列、R^3ベクトル |
| Q_m | 条件別のHouseholder直交反射 | 1 | R^{12×12}行列、Q_m^TQ_m=I |
| σ | 成分別ReLUまたはSiLU | 1 | R^12→R^12写像 |
| g_m | Mirror整合教師の解析的関数 | 1 | R^8→R^3写像 |
| p_m | 機能別の独立解析的MLP関数 | 1 | R^8→R^3写像 |
| γ | 独立機能成分の混合係数 | 1 | {0,0.35,1}、スカラー |
| y_m | 最終的な解析的教師ラベル（正規化前） | 1 | R^3ベクトル |
| β_m,u_m | 結合rank1の係数と単位方向 | 1 | 実スカラー、R^12単位ベクトル |
| S_m | 結合rank1の前後変換 | 1 | R^{12×12}行列 |
| NMSE | 分割内の教師出力変動で正規化したMSE | 1 | 非負スカラー |

Mirror整合解析関数を

\[g_m(x)=B Q_m^\top \sigma\{Q_m(Ax+b)\}+c\]

と定義し、異なる独立関数を混ぜることで

\[y_m(x)=(1-\gamma)g_m(x)+\gamma p_m(x)\]

という三世界を用意した。出力には共通の成分別アフィン正規化を適用する。\(\gamma=0\) はMirror生成の**意図的な陽性対照**であり、一般的な関数におけるMirror特有の優位を示さない。\(\gamma=.35\) は65%と35%の**係数での関数混合**であり、機能分散の35%を意味しない。\(\gamma=1\) は独立関数世界。主要評価は\(\gamma=.35\)・ReLUとした。

直交変換Qは12×12、Ax+bは12次元、Bの出力は3次元で一致し、各式はSI無次元。結合rank1は \(S_m=I+\beta_m u_mu_m^\top\) を前後に適用し、\(\beta_m=-2\) でMirrorと**完全一致**する。これにより、最適化による偏り／制約の利点を、表現クラスの優位と区別した。

## 2. 実装・固定プロトコル

- **学習対象:** 7方式（無条件共有、FiLM、学習Mirror、学習結合rank1、同係数級の小独立MLP、上限幅の独立MLP、固定乱数Mirror）×3世界×2活性×5シード=210モデル。
- **主比較:** `mirror` 対 `film`, `coupled`, `ind_budget`。最小4/5 seedでMirror Fresh NMSEが**各対照より10%以上小さい**ときのみスクリーニングPASS。Freshによる方式・学習率選択なし。数値閾値は探索seed707で非飽和を確認してから固定。対照誤差の最小可能値0に関して、今回の閾値は到達可能。
- 入力8次元、隠れ12、出力3、条件3。全方式に条件IDを明示し**外部ルータとMoEなし**、各例は1Viewだけ計算。教師の重み・反射方向は学生に渡さない。全方式で同じ訓練入力・ラベル・バッチ順を利用。
- 各条件512Train / 256DEV / 512Fresh / 512OOD、別seedの生成。Train各方式500 AdamW更新、batch128、lr0.006、weight_decay1e-4、DEV100更新間隔で最良checkpoint選択。OODは入力振幅を1.5倍。
- DEV-only探索seed707を訓練前に実施し、本試験の5 seedは101・211・307・401・503に固定。使用サンプルは新しいseedであるが、複数世界は独立の教師の生成分布を共有する設計であり、外部自然データではない。
- 全210 checkpointのハッシュを記録したSEAL.json（`fresh_generated=false`）の保存後のみ、Fresh/OOD生成と測定を実行。後から閾値、学習率、主世界・主活性は変更せず。

## 3. Freshの主比較：ReLU

**5 seed中央値、NMSE小さいほどよい。**

| モデル | 完全整合 γ=0 | 65%整合+35%独立 γ=.35 | 非整合 γ=1 |
|---|---:|---:|---:|
| 共有 | 0.20588 | 0.48500 | 0.74877 |
| FiLM | 0.09755 | **0.13338** | 0.15417 |
| **学習Mirror** | **0.07197** | 0.13486 | 0.23038 |
| 結合rank1 | 0.07515 | 0.13445 | 0.23741 |
| 独立幅5（同係数級） | 0.09874 | **0.12323** | **0.10683** |
| 独立幅12（容量上限） | 0.02654 | 0.03730 | 0.02356 |
| 固定Mirror | 0.11140 | 0.16308 | 0.32052 |

**主世界の対応seedごとのFresh NMSE比 Mirror/対照（1未満がMirror有利）:**

| seed | FiLM | 結合rank1 | 小独立MLP | 3対照全てに10%以上勝つか |
|---:|---:|---:|---:|---|
| 101 | 1.3111 | 1.1540 | 1.0934 | FAIL |
| 211 | 1.2196 | 1.0230 | 1.0720 | FAIL |
| 307 | 1.1662 | 1.0031 | 1.0944 | FAIL |
| 401 | 1.1251 | 1.0500 | 1.1683 | FAIL |
| 503 | 0.9173 | 1.0221 | 1.1490 | FAIL |

**事前登録の主判定は0/5でFAIL。** 第三者の未使用seedによる追加10本は、このゲート不通過のため実施しない。

### 構造が一致するときにだけ利益が出た

- **整合世界γ=0、ReLU:** Mirror/FiLM対応幾何平均比 **0.6583**（約34%の相対誤差減少）、5/5 seedで10%以上改善。Mirror/同係数級独立比 **0.6367**、5/5 seed改善。
- **整合世界γ=0、SiLU:** Mirror/FiLM **0.7311**、Mirror/小独立 **0.5622**（どちらも5/5 seed改善）。
- しかし、整合世界ですらMirror/結合rank1はReLU **0.9946**、SiLU **0.9837**。全seedで10%以上改善した例は0。**Mirror固有の上位表現力は存在しない**という厳密な包含関係と整合。
- **独立世界γ=1、ReLU:** Mirror/FiLM **1.4235**、Mirror/小独立 **2.1937**、いずれもMirrorに不利。SiLUも対応する比は **1.6907**、**2.0116**で不利。
- **部分整合γ=.35、SiLU:** Mirror/FiLM比 **1.0827**、Mirror/結合rank1比 **0.9958**、Mirror/小独立比 **1.1185**。改善が消えた。

これらは、共有できる対称性がある関数族にMirrorが適合する、しかし一般の機能を自動的に分解して効率化するわけではないことを示唆する。γ=0の教師はMirror形式で生成しているため、**設計上の有利さを除去した独立検証ではない**。

## 4. 機能選択の因果的な確認

同一入力で条件mを循環的に変更すると、MirrorのFresh NMSE増加中央値は、ReLUの整合世界で **+0.441**、部分整合世界で **+1.113**。固定MirrorやFiLMでも条件交換に応じた損失が生じるため、これは「Mirrorが異なる条件付き機能を学んだ」証拠だが、Mirror固有の優位の証拠ではない。**正解の条件IDを与えた実験**なので、実運用でRouterを学習できることも示さない。

固定ランダムMirrorが整合世界で学習Mirrorより悪い（ReLU 0.1114 対 0.0720）ことは、方向コードを学習することに一定の価値があるという示唆。ただしこれは小型合成関数・5 seedの比較であり、普遍的な効果ではない。

## 5. OOD・保存量・実CPU時間

**主部分整合世界ReLU、5 seedの中央値:**

| 方式 | OOD NMSE ↓ | 学習係数 | 実torch.save推論packet bytes | Batch128前向き中央値 μs |
|---|---:|---:|---:|---:|
| 共有 | 0.50075 | 147 | 2,973 | 12.7 |
| FiLM | **0.13624** | 183 | 3,353 | **25.2** |
| Mirror | 0.14818 | 183 | 3,353 | 68.4 |
| 結合rank1 | 0.14353 | 186 | 3,605 | 65.3 |
| 独立幅5 | 0.14469 | 189 | **3,101** | 50.3 |
| 独立幅12 | 0.04050 | 441 | 4,125 | 61.9 |
| 固定Mirror | 0.17006 | 147学習／183保存相当 | 3,353 | 69.3 |

- PyTorch 2.10.0+cpu、Python 3.13.5、NumPy 2.3.5、AMD EPYC 9V74 CPU。FP32、PyTorch 1スレッド。CPU clock固定なし。
- 前向き計時は各試行 batch128、25 warmup、5ブロック×80 forward。各モデル内中央値、世界ごと5 seed中央値。eager Python上の実測であり、最適化済みGPUへの外挿不可。
- 各学生500×128=64,000例提示、210モデル合計 **13,440,000例提示**。これは独立モデル間で合計した数で、同じネットワークが一度に学習した量ではない。訓練全体の壁時計 **92.13秒**。
- 同更新・同バッチ条件だが、MirrorとFiLMの反射／要素別ゲートは実演算量が異なる。**同学習FLOPsとは主張しない。** 学習係数数が近くてもシリアライズ時のモデルtensor数・メタデータでbytesが異なり、今回独立幅5のpacketはMirrorより小さい。

## 6. 不確かさ U

ペアログ比の標本SD/√5を合成標準不確かさの暫定記述値 u_c と定義（5個の教師seedの独立性を仮定）。k=2は記述的範囲であり、母集団の95% CIではない。

| 比較 | 対応Fresh NMSE比の幾何平均 | u_c（対数尺度） | k=2の比率範囲・記述的 |
|---|---:|---:|---|
| Mirror/FiLM, ReLU γ=0 | 0.6583 | 0.0555 | 0.5892–0.7355 |
| Mirror/結合rank1, ReLU γ=0 | 0.9946 | 0.0109 | 0.9731–1.0164 |
| Mirror/FiLM, ReLU γ=.35 | 1.1399 | 0.0600 | 1.0110–1.2853 |
| Mirror/結合rank1, ReLU γ=.35 | 1.0491 | 0.0249 | 0.9981–1.1027 |

最大の不確かさは教師を生成する関数族自体の偏り、ハイパラ探索の範囲、学習と評価分布、5 seedという小標本、initコードの初期条件、CPUスケジューリング。FiLMとMirrorはReLUで実際の初期View差が近かったが、SiLUでは初期View差が揃っていないため、SiLUの比較を初期化統制の因果実験とは呼ばない。

## 7. ERROR CHECK / インテグリティ

- 事前登録ソース・テスト・プロトコルを、学習開始前に別GitHub研究ブランチで固定。ローカルpreregコミット `67528ac`。凍結ファイル8点のGit blob SHAをremoteから再読して一致を確認。
- 初期の単体機構テスト **7/7 PASS**。完全Mirror整合の教師重み/正しいコードを設定したoracleはテストNMSE<1e-10。**oracleには教師構造を与えているため、学習成功の証拠としない。**
- 学習210/210終了。`results/SEAL.json` SHA256 `71705848e976b5aa4fe4ede43ec0ba6225239385a0d033ffbcd757714bad4a65`。Manifestの `fresh_generated=false` を保持。
- 全checkpointのSHA256照合210/210。別再生210/210。最大Fresh/OOD/DEV絶対NMSE誤差**0.0**。プロトコルSHA256 `e07b84f94d2829821439ac39e82edf558e25d3311c885d961adf968e54f61b89`。
- 主試験の一切のFAIL/PASS条件、コード・出力変数はFresh後に変更していない。ローカルアーカイブに全210チェックポイントと結果、source、tests、pilot、封印ログを同梱する。

## 8. H/T/D/C：次に何を検証すべきか

**H:** 同じ基本非線形関数の座標変換で表せる部分だけをMirrorで共有し、**少数のprivate低ランク残差**を別に保持すれば、部分整合世界における性能低下を防ぎ、FiLM/結合rank1/独立より良い実保存量・性能のPareto点を作れるか。

**T:** 事後探索の補助（別seedでγを細かく掃引）を主試験へ混ぜない。まずPrivate容量・分布別の非飽和課題を独立DEVで定め、別のprereg登録でMirror＋privateと同バイトFiLM＋private、同バイト結合rank1＋privateを訓練する。学習プロセスの段階（既知核／未知コード／全共同学習）も切り分ける。

**D:** Mirror＋privateが同品質かつ低保存量または同保存量かつ低Fresh/OOD誤差を達成し、実時間悪化が規定範囲ならPASS。基本Mirrorが既に含まれる結合rank1との競争に勝てなければ、Mirror固有の採用根拠はない。

**C:** 一般の低ランク残差やPrivate小MLPの単純な追加で利益が説明でき、Mirror側に固有利益がない可能性。多くの関数が非共通なら無理な共有は負の転移になる。

関連分野は幾何学的表現学習（共役変換・錐）、統計学的マルチタスク学習（共有／privateのバイアス・分散）、機構的解釈可能性（条件交換介入）、計算機システム（実バイトと前向きコスト）。

**関連一次文献:** Yuanら, Householder Reflection Adaptation (NeurIPS 2024, https://arxiv.org/abs/2405.17484)、Fu & Cohen, Conic Activation Functions (PMLR 285, https://proceedings.mlr.press/v285/fu24a.html)、Wenら, BatchEnsemble (ICLR 2020, https://arxiv.org/abs/2002.06715)。

**結論:** **Mirrorは「関連する関数の座標変換」に対して役立つが、既存の結合rank1より一般に優位とは言えず、無関係関数の自動分離も成功していない。** 価値のある次の問いは、Mirrorがどれだけの機能を共有側へ回収し、どれだけをprivate側へ残すべきかという資源配分である。

---

## Appendix A — 事後探索：混合係数γの連続掃引（主判定に不採用）

## Fixed-grid results (ReLU, five independent teacher seeds)

| γ | Mirror / FiLM Fresh NMSE (paired geometric mean) | Mirror / coupled-rank1 | Mirror / small independent | Mirror ≥10% better than FiLM |
|---:|---:|---:|---:|---:|
| 0.00 | 0.7047 | 1.0187 | 0.5840 | 4/5 |
| 0.10 | 0.7629 | 0.9924 | 0.6642 | 4/5 |
| 0.20 | 0.9412 | 1.0221 | 0.9253 | 2/5 |
| 0.30 | 1.1267 | 1.0504 | 1.1826 | 0/5 |
| 0.40 | 1.1877 | 1.0379 | 1.2400 | 0/5 |
| 0.60 | 1.2386 | 0.9842 | 1.3941 | 0/5 |
| 1.00 | 1.4092 | 1.0396 | 2.1080 | 0/5 |

**Observation:** Mirror's estimated paired advantage versus FiLM is present at γ=0 and γ=0.1 (ratio below1), uncertain near γ=0.2, and reverses near γ=0.3 and thereafter. This pattern is only a posthoc diagnostic, not an independently preregistered Mirror superiority claim. The coupled rank1 model is almost equal to Mirror throughout the sweep, consistent with exact representation inclusion.

**Uncertainty:** At γ=0.2 the geometric mean Mirror/FiLM ratio is 0.9412, paired log standard error u_c≈0.0743, and a descriptive k=2 interval corresponds to ratios 0.811–1.092; the benefit is genuinely uncertain with five seeds. At γ=0.3 the ratio is 1.1267, u_c≈0.0382, descriptive range 1.044–1.216. Neither interval is a calibrated 95% CI. At γ=0.1 the paired ratio is 0.7629, u_c≈0.1286, interval 0.590–0.987.

**Interpretation:** A plausible conditional design principle is to reserve Mirror for a geometrically coherent shared component and represent the residual with a separate small private mechanism. Next experiment must compare a Mirror+private model with equally budgeted FiLM+private and coupled-rank1+private, not a crippled pure FiLM.

**Verification:** This appendix was planned after inspecting MA-1183 main Fresh; training code/data protocol for this *separate sweep* was committed locally before running new sweep seeds. `SWEEP_TRAIN.log`, `SWEEP_SCORE.log`, `SWEEP_VERIFY.log`, `SEAL.json`, `VERIFICATION.json` included. Main MA-1183 frozen gate remains FAIL 0/5.
