# MA-1183後の第一原理研究計画 — Mirrorの幾何学的適用領域とShared/Private設計

更新日: 2026-10-30。対象ブランチ: `research/ma-1183-native-function-sharing-20261030`。
**この文書は次の研究提案であり、新規実験の事前登録や成功主張ではない。** MA-1179–1183のFAILを変えない。`main`へマージしない。

## 1. 根拠が確定した事項・未解決事項

- **事実（MA-1183）**：ReLU部分整合世界（独立関数の線形混合係数 γ=.35）で、MirrorがFiLM・結合rank1・同係数級独立MLP全てをFresh NMSEで10%以上改善する主ゲートは0/5 FAIL。5seed、210学習済み学生、全210再生一致。
- **事実（陽性対照）**：γ=0で教師をHouseholder構造から生成した場合、MirrorはFiLMや小独立MLPより低誤差になりやすい。しかしMirrorを厳密に含む結合rank1との優劣は立証できない。したがって自然関数の圧縮可能性や一般優位の証拠ではない。
- **探索（主判定に使わない）**：別の5seedでγ=.1/.2/.3の対応Mirror/FiLM Fresh NMSE幾何平均比は .7629/.9412/1.1267。γは出力分散に占めるprivateの割合ではない。普遍的な遷移点も未確証。
- **事実（MA-1182）**：主タスクのFiLM Fresh CE ≈ .00022に対して絶対改善 .02 nat という主ゲートは数学的に到達不能。今後、非飽和確認が済むまで主比較を始めない。RoPE/NoPEの役割分担とMirror固有の利益は別。
- **未解決**：共有できる関数の特徴を、教師がMirrorで生成されていない場合に学習前に診断できるか。幾何学的に共有可能でも、joint optimizationで獲得できるか。Private残差とMirrorの組合せが、単純なFiLM/結合rank1+Privateに実効的に勝つか。

## 2. 第一原理上の重要な境界

**前提**：特徴量、重み、NMSEを無次元量として扱う。コードは各条件に別途保存し、選択器・推論計算・状態は無料扱いしない。

| 記号 | 意味・定義 | SI単位 | 定義域／前提 | 型 |
|---|---|---|---|---|
| d | 共有隠れ次元 | 1 | 正整数 | スカラー |
| K | 条件別機能数 | 1 | 正整数 | スカラー |
| x | 観測入力 | 1 | R^p | ベクトル |
| z | 共有隠れ前活性 | 1 | R^d | ベクトル |
| Q_m | 第m条件の直交Mirror | 1 | R^{d×d}, Q_m^T Q_m=I | 行列 |
| D_m(z) | ReLUの活性マスク | 1 | 対角要素0/1、非境界入力 | 行列 |
| J_m(z) | Mirror活性の局所Jacobian | 1 | R^{d×d} | 行列 |
| u_m | 反射方向 | 1 | R^d, 長さ1 | ベクトル |
| β_m | 結合rank1の係数 | 1 | 実数 | スカラー |
| S_m | 結合rank1の変換 | 1 | R^{d×d} | 行列 |
| E_S,E_M,E_I | shared-only／Mirror oracle／独立上限モデルの評価誤差 | 1 | 非負の実数 | スカラー |
| ρ_orbit | 共有可能な改善量の診断比 | 1 | E_S>E_Iのとき定義 | スカラー |

1. ReLU境界を持つ単純な座標変換活性における局所的な作用は `J_m(z)=Q_m^T D_m(z) Q_m`。このJacobianは（境界を除き）対称・冪等で、固有値は0/1。ただし一般のMLP上下射影を含めた全Jacobianは射影行列とは限らない。
2. 同じQでReLU射影を繰り返すだけなら冪等で、新しい情報処理が増えない。Loopの利得には途中のAttention、Residual、View変更などが必要。
3. Householder `Q_m=I−2u_m u_m^T` は結合rank1 `S_m=I+β_m u_m u_m^T` の β_m=−2 の特例。優位性は表現力の包含ではなく、同実コストでの**帰納バイアス、学習安定性、OOD、記述長**でしか示せない。
4. Oracleの機能空間評価で「共有すれば改善可能な分の何割をMirrorで回収するか」を診断する一例は

   `ρ_orbit=(E_S−E_M)/(E_S−E_I)`（E_S>E_Iのとき）。

   ρ_orbit は一般に0～1へ制限されない記述量であり、推定器・最適化誤差に依存する。値が高いことは学習容易性やコスト優位の証明にはならない。各Eの訓練・検証データ分割を厳密に分ける。
5. **次元検査**：Q、D、J、Sは d×d。損失Eの比ρは無次元。`Q^T D Q`は d×d。演算は整合。

## 3. 優先順位の提案

| 順位 | 次の研究 | 目的／前提 | 最初の停止条件 |
|---|---|---|---|
| **P0** | 独立に作られた関数に対する**Mirror-orbit適合度検査** | MLPの教師をMirrorで生成しない。固定共有核と既知/未知コードを分け、実際の出力・Jacobian介入でMirrorがどの範囲を近似できるか計測。FiLM/結合rank1/独立を同時測定 | 構成上有利な教師以外では、同費用FiLMと結合rank1に対する機能空間・OODの利得なしなら、当該教師族への展開を止める |
| **P1** | **Shared Mirror + 小Private残差**（最初の性能仮説） | P0で共有可能な関数族を見付けた後、Private rank 0/1/2/4をMirrorだけでなくFiLM・結合rank1にも同等に与える。別seed Fresh/OOD、実bytes、学習FLOPs、推論時間を比較 | Mirror+privateが最強非Mirror対照の品質/保存量/時間Pareto前線を越えなければ不採用 |
| P2 | **MoEなしのMirror-Loop計算選択** | P0/P1が示した機能差を使い、Loopの前後で別Viewに切替。Wuらの学習線形J、同費用FiLM/low-rank、無変調Loopを対照 | View切替による因果的機能選択・OOD利益がなければ、Loop+MoEの増設を停止 |
| P3 | 自然データ・小型実LMへの移行 | 独立タスク、入力だけのView選択、バックボーン異種seed、長さ汎化、KV品質 | 合成関数のみの改善なら自然LMの優位を宣言しない |

## 4. P0→P1を進める新規実験仕様案（**未凍結**）

**H（反証可能）**：非Mirror生成の関連タスクで、Mirrorが必要な共通幾何を低記述長のViewコードとして回収でき、少量private残差を加えると、同保存量・同学習計算量のFiLM+privateと、Mirrorを厳密に包含するcoupled-rank1+privateをFresh/OODで上回る。

**T（最小検証）**：
- まず教師をMirrorで生成しない関連／無関連／共起・疎性を独立操作した新規関数族を3つ以上用意。MA-1183の教師・評価seedは再使用しない。最初に固定核oracle fit（関数族の適合度）、次にコードだけ学習、最後に核+コードjoint学習を別実験で評価。
- Private rank=0,1,2,4を探索。全アームが同じ実byte上限・同じ総学習FLOPs予算を使う。少なくともshared-only、FiLM+private、coupled-rank1+private、Mirror+private、private-only、小独立上限を比較。
- 最初はoracle機能IDを全方式に等しく与える。入力からIDを推定するルータは**追加の別研究**で、費用と情報漏洩を必ず計上。
- 開発seed最低5、確認seed最低10。さらに新しい関数族そのものを独立Freshとして封印し、乱数だけの再分割で汎化したとは呼ばない。
- 先行する**別seedのパイロット**で、最強非Mirror対照のDEV品質が飽和しておらず、主改善閾値が数学的に達成可能かを点検。後からFreshを見て閾値・アーム・課題を変えない。

**D（判定）**：
- 一次：独立確認のペアFresh NMSE比 `Mirror+private / min(FiLM+private, coupled-rank1+private, private-only, independent-budget)` の幾何平均 ≤0.90（候補、正式登録時固定）。対照ごとの対応比較とseedごとの勝敗も報告。
- 二次：OOD劣化が規定許容内、実serialized inference bytesが比較群と≤5%差、総FLOPsが規定許容内、CPUまたはGPUの実測レイテンシを必須報告。独立Expert上限は必ずしも同byteでないため分類する。
- 途中で「0.02 CE改善」のような基準が対照誤差の非負下限を超えたら、事前PILOT段階で課題を変更する。Fresh後に変更しない。

**C（破れ方）**：非Mirror教師の変換規則が非直交／非共役、一般coupled rank1が同じ機能をよりよく学習、privateがほぼ全効果を説明、複数Viewが同一機能へ収束、MoE的な本物の新しい計算核が必要。

**U（不確かさ）**：教師生成分布、データ相関・スパース率、seed、学習率/初期化、Private保存量、Python eager vs fused GPU、対照最適化予算。基準分布を固定して対応ログ比から `u_c=SD(log ratio)/sqrt(n)` と記述的に報告。k=2範囲は少数seedでは95% CIと呼ばず、確認時はbootstrapなど頑健な方法も検討する。

**最重要の実施順序**：P0の構造適合度→P1のPrivateとの分解→P2のLoop実装。コードだけ増やす探索や、簡単な合成分類の再実行をしない。すべての予算とFAIL結果を保持する。

## 5. 関連分野と一次研究

- 幾何学/数値解析：Householder直交変換と共役作用、錐へのReLU射影、Jacobianの同定。
- マルチタスク学習/統計：共有成分・private残差のモデル選択、負の転移、少数seedの不確かさ。
- 機構的解釈可能性：View交換・遮断・activation patchingによる計算の因果選択。
- 計算機システム：アクティブFLOPs、メモリ、ルータ負荷、反射の融合実装。

原論文: Yuan et al., NeurIPS 2024 HRA [arXiv:2405.17484](https://arxiv.org/abs/2405.17484); Fu & Cohen, PMLR 2024 [Conic Activation Functions](https://proceedings.mlr.press/v285/fu24a.html); Wen et al., ICLR 2020 [BatchEnsemble](https://arxiv.org/abs/2002.06715); Wu et al., 2026 [Shared Weights, Selected Computations](https://arxiv.org/abs/2609.39892); Cammarata, 2026 [Neural Networks Are Already Clean](https://already-clean.goodfire.com/)。

## ERROR CHECK

- この文書の数値はMA-1183本試験および別区画の事後探索報告から転記したもので、新しい訓練結果ではない。
- Householderはcoupled-rank1に厳密に含まれる点を全候補で維持。
- MA-1182の不可能な判定閾値を明記し、同種の設計ミスを防ぐ。
- Repoの次候補に新しいMA IDは予約していない。正式にプロトコルを凍結する直前にlive registryから採番する。

## 2026-10-31 update — MA-1184の正式な研究候補登録

この文書の2026-10-30時点での「MA IDは未採番」という記述は**当日の歴史的記録**。2026-10-31に現在の研究ブランチでレジストリ最大ID 1183を再確認し、次の一件だけを **[MA-1184（UNTESTED）](../../experiments/mirror_applications/ma-1184-natural-function-orbit-diagnostic/README.md)** として正式に登録した。

- P0: 独立生成・独立訓練済みでHouseholder変換由来ではない教師関数の、機能空間でのMirror-orbit適合度。既知の関数構築を陽性対照から自然な関数へ一般化できるか測る。
- P0の判定では、固定共有核の関数適合度、コード学習、核とコードの共同学習を別々に評価。FiLM、Mirrorを包含するcoupled-rank1、BatchEnsemble、独立MLPを同資源基準で比較する。
- **P1のMirror+Privateは未採番・未訓練**。P0で教師機能の適合領域が見つかるまでは拡張しない。
- [実験計画](../../experiments/mirror_applications/ma-1184-natural-function-orbit-diagnostic/EXPERIMENT_PLAN_JA.md)、[未凍結プロトコル](../../experiments/mirror_applications/ma-1184-natural-function-orbit-diagnostic/PROTOCOL_DRAFT.json)。本更新は実験結果でも事前登録済み学習でもない。
- MA-1179〜1183の負の記録と探索／確証の区別を保持。mainへのマージは行わない。
