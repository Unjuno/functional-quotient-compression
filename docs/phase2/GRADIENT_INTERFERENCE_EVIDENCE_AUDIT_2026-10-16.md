# Mirror 共有学習における勾配干渉：成功・失敗の証拠監査と次の切り分け（2026-10-16）

**状態: EVIDENCE AUDIT / PROPOSAL ONLY。新規学習・追加実験・正式MA判定は行っていない。main、台帳、凍結済み実験条件は変更しない。**

基準ブランチ: `research/adaptive-mirror-core-cpu-20261014`（このファイルの親）。その `IDEA_REGISTRY.csv` は 1155件：**54 PROMISING / 132 FAIL / 969 UNTESTED**。PROMISING は採用確定でも自然言語LLMの容量証明でもない。2026-10-06 時点の main にある MN 統合文書は過去の評価として併読する。

## 結論（事実・推論・仮説を分離）

- **事実**：過去の一部の共有モデルは勾配干渉を明示的に抑制せずに成功した。一方、共有方式に明確な失敗もある。
- **事実**：2026-10-13 の [活性化共有方式メモ](MIRROR_ACTIVATION_SHARING_METHOD_INTAKE_2026-10-13.md) C3/D5 には勾配・Fisher・関数出力差を使った共有先選択が研究候補として登録済み。MA-836（NTK直交）と MA-844（Fisher幾何）は UNTESTED。従って「全く検討していなかった」は不正確だが、**体系的な勾配干渉の実測と介入による因果同定は主要記録にない**。
- **推論**：共有の悪化には (I) 勾配干渉、(II) 関数表現族の不足、(III) ルータの誤選択、(IV) 実行/合成構造の欠落、(V) サイズ/実装オーバーヘッド、(VI) 対称変換の無効作用が混在する。
- **仮説**：勾配干渉を検知し、共有曲線・Mirror座標・専用残差への容量を双方向に再配分できれば、固定共有より良い品質/保存バイト前線を実現できる条件がある。まだ実証されていない。

## 1. 成功した例（干渉対策を明示せず、成功した範囲だけを報告）

| 実験 | 記録された成績 | 確定すること / 確定しないこと |
|---|---|---|
| MN008 | 明示的な再結合教師信号により held-out再結合精度が63.12%→89.56%（元world）。ただし additive shared-base のNLLが強い | 共有因子は学習できる。Mirror変換固有の優位性はない |
| MN009 | 86,127 B Mirror が85,665 B の狭い独立soft-MoEよりIID/OOD NLLで3/3 seed優勢 | 広い共有経路に有利な場面あり。Dense/simple gate への一般優位性なし |
| SRM001 | private32の構造化count課題、1000更新：86,166 B Mirror rank16で99.20%、88,215 B MoEで82.89%、4/4 worlds方向一致 | 明示的に分解可能な規則で共有が学習効率上有利。長時間収束/自然課題での一般優位性は未証明 |
| SRM002 | 非可換順序の **供給されたexecutor** と符号付き共有基底でmedian NMSE 5.94e-14。Mirror stretch/shearは2.91e-2 | 適切な基底+演算順序の供給は強力。Mirror幾何の優位性はむしろ否定的 |
| MS008/MS012 | 観測校正→sensor ID非入力の共有core→必要な残差View が有効。MS012 learn→prune はprune-from-startにseen 7/10、transfer 8/10勝利 | 分離/残差の有用性。勾配衝突原因の直接証明ではない |
| MA-241 | 回転Viewと整合する合成教師で3/3 freshの品質/容量ゲートPASS、24,286 B vs untied 45,681 B。ただしeager推論が遅い | 既知の低次元関数軌道を表現できる場合の学習可能性 |
| MA-268 | 回転整合合成タスクの学習済み活性Viewで3/3 fresh quality/byteゲートPASS、4,167 B vs IA3 4,273 B、独立 7,933 B。MACはIA3の1.5倍、eager速度不利 | 活性調整が成功する領域はある。実用速度/自然課題は未証明 |
| MA-255 | **post-fit** 整合教師では734 Bでfresh近零MSE。一方、別の固定1200更新プロトコルは開発品質ゲートFAIL | 表現可能性と学習到達性を分ける重要事例。ただし同一実験条件の介入比較ではない |

出典: [MN007–010](RESEARCH_STATE_THROUGH_MN010.md), [SRM001](SRM001_SHARED_RULE_MOE.md), [SRM002](SRM002_NONCOMMUTATIVE_COMPOSITION.md), [MS008](MS008_VIEW_DECOMPOSITION.md), [MS009–014](MS009_MS014_VIEW_OPTIMIZATION.md), [MA-241報告](../../experiments/mirror_applications/ma-241-expert-tying-mirror/README.md), [最新MA CLAIM_LEDGER](../../experiments/mirror_applications/CLAIM_LEDGER.csv)。

**解釈上の注意:** 上の「成功」は別々の判定基準（機構・性能・省バイト・学習効率）であり、単一の総合成功率として足し合わせない。MA-255/268/314等の結果は台帳の異なる protocol variants を混ぜない。

## 2. 失敗を勾配干渉に帰属してはいけない例

| ケース | 観測 | より直接的な候補要因 / 現状 |
|---|---|---|
| Phase I FQC 28M/8M/1M | post-training sharing/private が強い非共有codecを上回らない | **事後圧縮/表現の問題**。共同学習時の勾配相殺を直接の原因にできない |
| MT002B / MT003 | oracle routingでも Mirror OOD accuracy 約12.5%/15.0% vs independent MoE 53.9%/75.3%。gain幅 sweep でも独立MoEが優勢 | **表現自由度不足**。正しいルータだけでも救えない |
| RF1 / MA-333 | 多数の回転Viewは安定なDense性能優位なし。正確な符号/伸縮対称は同じ関数 | **新しい関数方向がない**。勾配手術で存在しない自由度は作れない |
| SRM003 | 24/24 atomic規則を正確保持。直接2規則合成は6400更新で約13–19%、同じ模型を外部順序で2回呼ぶと100% | **順序を持つexecutorの不在**が強い候補。損失伝播/表現も未分離 |
| MA-199 | 共有勾配平面＋角度の提案は1103 B vs LoRA1196 Bで固定90% byteゲート未達。一般共有係数はほぼ同品質 | **バイト/最適化状態のオーバーヘッド**。勾配干渉治療実験ではない |
| MA-368 | heldout width×depthでMirrorはnative supernet・scalar controlに両seed劣後 | **作用位置や制御形状が不適切**な可能性 |
| MA-407 | Givensの行ノルムに対するdemodulationが実際に不変。rawとdemodの精度/CVが一致 | **構造的null**。勾配問題ではない |

出典: [2026-10-06統合記録](RESEARCH_STATE_2026-10-06.md), [SRM003](SRM003_CAUSAL_DISCOVERY.md), [MA-199](../../experiments/mirror_applications/ma-199-gradient-coordinate/README.md), [MA-368](../../experiments/mirror_applications/ma-368-factorized-width-depth-mirror/README.md), [MA-407](../../experiments/mirror_applications/ma-407-demodulated-mirror-moe/STATUS.md)。

## 3. とくに勾配干渉の介入検証を優先すべき例

1. **MA-255:** post-fit と end-to-end学習のギャップ。ただし異なるprotocolであるため、同じ固定teacher/容量条件で「oracle fit→同時最適化→PCGrad→選択的private」を作り直す。
2. **MN007/MT001:** 自発的factor分離と学習品質のギャップ。MT001は高いgradient agreementと低いtask accuracyが両立したため、**勾配整合だけでは十分条件でない**。
3. **AMA-001（2026-10-14）:** 120モデル/4条件/3開発seed。partial taskで活性化+privateはprivateなしに対して3/3改善したが、nativeの品質・バイト・CPU実行速度ゲートFAIL。mirror+private fixed GELUの平均partial nMSEは0.0637、learned activation+privateは0.0770、native multiheadは0.0893（partialのみ）。unrelated ではnative multihead 0.0877に対しlearned activation+private 0.2761。**曲線学習とprivate追加だけでは十分でない**。Fresh 5201–5205は未開封。
4. **MA-314/315:** aligned taskへの小座標は有望だが、混合/独立taskとstrict byte gateで崩れる。容量配分の必要性とMirror固有overheadを分ける。

参考: [活性化共有方式メモ](MIRROR_ACTIVATION_SHARING_METHOD_INTAKE_2026-10-13.md), [AMA-001](../../experiments/mirror_applications/research_intake/adaptive_mirror_core_20261014/README.md), [MA台帳](../../experiments/mirror_applications/IDEA_REGISTRY.csv)。

## 4. 勾配干渉を測る定義（未測定の提案）

タスク別損失の**共有部分のみ**で得たパラメータ勾配について、

```math
g_t=\nabla_{\theta_s} L_t,\qquad
c_{ab}=\frac{g_a^{\mathsf T}g_b}{\|g_a\|_2\|g_b\|_2},\qquad
R=\frac{\|\sum_t p_tg_t\|_2}{\sum_t p_t\|g_t\|_2}.
```

| 記号 | 日本語の意味 | SI単位 | 定義 | 定義域/前提 | 型 |
|---|---|---|---|---|---|
| \(\theta_s\) | 共有パラメータ | 1（規格化座標） | 対象共有blockをflatten | \(\mathbb R^d\), 要素尺度を固定 | ベクトル |
| \(L_t\) | タスク別損失 | 1 | 同一規格化の平均CE/MSE | 非負、微分可能 | スカラー |
| \(g_t\) | 共有blockのタスク勾配 | 1 | \(\nabla_{\theta_s}L_t\) | \(\mathbb R^d\), optimizerのweight decay除外 | ベクトル |
| \(c_{ab}\) | 2タスクの勾配余弦 | 1 | 内積/ノルム積 | \([-1,1]\), 両勾配ノルムが非零 | スカラー |
| \(p_t\) | タスク重み | 1 | 訓練mix比率 | \(p_t\ge0,\sum_t p_t=1\) | スカラー |
| \(R\) | 混合勾配の相殺比 | 1 | 合成勾配norm/個別normの加重和 | \([0,1]\), 分母>0 | スカラー |
| \(a,b,t,d\) | タスク番号、勾配次元 | 1 | 整数index | \(d\ge1\) | 整数スカラー |

**次元チェック:** 損失と規格化パラメータは無次元ゆえ勾配も無次元。c, R はノルム比で無次元。ゼロ勾配時は cosine をNAにし、ノルムとstationarityを別報告する。

重要な限界:
- \(c<0\) は必ず「学習できない」ではない。完全な反平行等では一方を下げる方向が他方を上げる。\(R\ll1\) は相殺の兆候だが、Pareto定常点・学習率・停滞・データノイズを区別しない。
- 生の勾配余弦は再パラメータ化に依存する。**同じ共有tensor、同じ規格化**で比較し、関数出力差、タスク別oracle fit、Jacobian rank、Fisher/NTK系の補助統計も報告する。
- 同じ入力・同じtask mix・同じ更新時点で記録し、all-layer平均だけでなく layer/FFN/activation係数/座標/private別を保存する。
- 候補が関数的な零方向なら \(g\) を加工しても有用な機能差は増えない。

## 5. 次の最小介入実験（実施していない）

**H1（診断）:** aligned / partially-private / unrelated と ordered-rule で、低性能と負勾配干渉が共変するか。ただし共変だけでは因果でない。

**H2（介入）:** 勾配相殺が確認された同一共有blockで、PCGrad型の勾配投影、選択的private、併用を比較すると、相殺条件に限って最悪タスク品質を改善できるか。**等バイト・同一データ・同等更新/active compute**のFiLM/低ランク/native baselineを必須とする。

**H3（反証）:** Oracle fitでも届かない/関数的null/2-call executorでしか解けない例は勾配介入のみでは救えない。これをネガティブコントロールにする。

**T（最小検証）:**
- 先に解析的2-task conflictのunit test、次に少なくとも **5独立world × 3独立初期seed** のpaired検証。開発用worldで閾値決定→新規freshで凍結判定。既存AMA-001の5201–5205は新実験の調整に流用しない。
- 固定パラメータ予算で (A) shared、(B) shared+gradient surgery、(C) shared+gated private、(D) shared+gradient surgery+gated private、(E) shared+Mirror可変活性＋gated private、(F) native FiLM/LoRAを比較。Eの追加自由度は全保存量と評価時間に課金。
- 共有θの勾配衝突率、相殺比R、機能Jacobian、routing、最悪タスク損失、平均損失、actual serialized bytes、active MAC/FLOPs、CPU batch1/64、追加パラメータの実際の物理疎化を報告。
- primary/secondary gateはfresh実行前に固定する。無関係関数と構造的nullを反例条件に含める。

**D（提案ゲート、後付けでないよう将来凍結）:**
- **PASS:** 相殺の大きいworld群で、候補が最強の同バイトnative controlを少なくとも4/5独立worldで最悪タスク損失5%以上改善し、平均損失悪化1%以下、総バイト超過なし、batch64 CPU時間1.25倍以内。同時に相殺低下と機能差増大の介入追跡がある。
- **FAIL:** 同バイトnative controlに負ける、相殺を除いてもtask損失が改善しない、最悪タスクが悪化、または構造的null上でのみ改善するという誤帰属。
- **UNCERTAIN:** oracleに未到達・計測条件相違・fresh漏洩・world間符号が不一致・性能とruntimeがtradeoffで事前閾値に未達。5 worlds は大きな一般化を確定する統計規模ではない。world単位paired効果と区間を併記。

**C（対立仮説）:** 表現のJacobian rank不足、変換可換性、失われた順序構造、ルータ情報不足、教師familyへの設計適合、強いnative baselineの存在、CPU Python overhead。

**U（主要不確かさ）:** ミニバッチ推定勾配、タスクmix比率、勾配尺度、初期seed、teacher生成、soft-to-hard gate近似、物理checkpoint format、CPU clocks。平均とworst-taskの両方を保存し、world単位のpaired差と変動範囲を報告。現時点で観測由来の\(u_c\)や包含係数\(k\)を算出できるデータはないため、数値を捏造しない。

## 6. 実装順と研究上の境界

1. 既存の学習比較の**診断再実行**で gradient logging を追加（固定source/seedの整合を検証し、旧結果は書き換えない）。
2. 表現不足と勾配相殺を別々に起こす最小toyで、介入が働く/働かない境界を検証。
3. 正規化・勾配手術で足りるのか、共有度の動的split/mergeや専用スプラインが必要かを対照付きで判断する。
4. 勝てなければshared splineを強制導入しない。強いnativeの勾配手術/FiLM/LoRAが勝つこと自体も有用な負例として保存。

**学際的接続:** 多目的最適化（共通降下方向）、継続学習（負の転移/忘却）、情報幾何/NTK（勾配尺度と機能距離）、符号長最適化（専用化の保存コスト）、計算グラフ設計（中間状態と順序制御）。

**ERROR CHECK:** 未測定の勾配衝突を既知原因と記載していない。PROMISINGとPASS/ADOPTEDを混同していない。MA-255の異なる2プロトコルを同一試験の追試と扱っていない。AMA-001の部分ゲートPASSと総合FAIL_DEV_GATEを分離している。CPU合成タスクから自然言語LLM一般の性能を外挿していない。
