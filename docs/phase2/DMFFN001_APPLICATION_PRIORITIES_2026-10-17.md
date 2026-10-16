# DMFFN-001 応用戦略：Dual-Input Mirror SwiGLU（2026-10-17）

状態：**研究候補の順位付け。追加の応用実験は未実施。** 過去の MA / TM / GIC 実験判定は変更しない。

## 0. 科学的前提：何を観測したのか

共有された3つのSwiGLU重み \(W_g,W_u\in\mathbb R^{d\times h}\)、\(W_d\in\mathbb R^{h\times d}\) と二軸のMirrorを使用：

\[
G=xW_g,\quad U=xW_u,\qquad
F_{ij}(x)=\{\mathrm{SiLU}(G R^g_i)\odot(UR^u_j)\} W_d.
\]

K_g個のgate ViewとK_u個のup Viewから、*最大で* K_g×K_u通りのアドレスを構成できる。これはK_g×K_u個の独立な関数自由度や独立情報量を意味しない。独立した行列を9組保存する代わりに小さなコードを保存できるのは、対象関数群に**共有・因子化できる構造**がある場合である。

既存DMFFN-001（5独立世界×2 seed×4条件×12方式＝480本評価、別3世界×2 seed×3条件×6方式＝108長時間追補）では、3×3=9論理MLPで以下が成立：
- **同容量・同パラメータ比較：** Mirror12 / native gain12 / per-pair Mirror4 は648学習パラメータ、5,081 B。aligned教師でMirror12/gain12の世界対応MSE比中央値0.314、5/5世界でMirrorの方が良好。
- **組み合わせ汎化：** 9組中6組のみ学習し、3組を完全holdout。Mirror12 test NMSE中央値0.0680、native gain12 0.2594、per-pair Mirror4 0.4673。gate×upの因子分解が効いた。しかし教師はGivens Mirrorで構成したため、教師の構造との一致が強い。
- **限界：** 無関係な9関数ではMirror12 test NMSE 0.676、native gain12 0.590、9独立MLP 0.0199。自動拡張の品質＋容量ゲート0/5 FAIL。独立9MLPとの差があるので「同品質で77.7%圧縮」は未達。CPU eager batch64の9並列出力ではMirror12 0.1644 ms、独立9MLP 0.0814 msで、実時間の短縮はFAIL。
- **より長い学習：** 6,000更新でaligned NMSE Mirror12 0.0237、独立9MLP 0.0133。モデル表現オラクルは8/8教師世界で正確にゼロ誤差なので、この差はaligned条件の表現不能性の証拠ではない。
- **プロトコル上の問題：** 片側Mirrorの単独比較は未使用側の角度テンソルも保存する実装で、容量の公平比較には使用しない。今後の改訂実験で修正する。

[実験報告](../../experiments/mirror_applications/research_intake/dmffn001_dual_input_swiglu_20261017/REPORT_JA.md)、[固定条件](../../experiments/mirror_applications/research_intake/dmffn001_dual_input_swiglu_20261017/PROTOCOL.json)を参照。

## 1. 第1候補 P0 — 二軸合成型の疎MoE（gate × up）

**なぜ：** 長い期間にわたり一つの共有FFNを保持しながら、少数のgate/Upコードの直積で多数の論理expertをアドレスする。動的ルータは \((i,j)\) を一括で選んでも、二つの小ルータ \(i\sim p_g(\cdot|x),j\sim p_u(\cdot|x)\) から選んでもよい。後者は独立性の仮定を追加するので別モデルとして比較。

**必要な実験：** 通常の独立Top-1/Top-2 SwiGLU-MoE、shared-expert MoE、FiLM/IA3、rank-1/2 LoRA、native factorized gain、pair専用Mirror、shared-only、Mirror＋低rank private residual を同じ**実保存バイト・実active FLOPs・学習計算量**で比較。Router entropy/occupancy、worst-expert quality、rare routeと強い専門家のgradient cosine、認識した未学習組の汎化を評価。

**計算の注意：** 選択したペアが1個ならgate/up/downの重い行列積は一組分 \(3dh\) のまま。複数のペアを同じ入力で計算するとき初めて \(2dh+Pdh\)（Pはactive pair数）という大きな射影の再利用が可能。必ずMirror変換、選択ルータ、down、メモリ転送を課金する。独立MLPを9個すべて評価した場合 \(27dh\)、9個の合成Viewなら \(11dh\) のMAC主項だが、実CPU時間の優位は既存実験で**否定的**。小さなViewが9個の独立expertに等しいわけではない。

**合否：** native factorized gainとnative low-rankが同バイト・同active computeで匹敵するならMirror固有の利得FAIL。実データのルーティングとunseen pairの品質が揃って初めてPROMISING。

## 2. 第2候補 P0 — MTP／Speculative draft用の直積FFN

**構成：** 共有prefix/Transformerの最後にあるSwiGLUの **gate軸を未来位置 +1…+P**、**up軸を候補系列の潜在plan/branch** に対応させる。1回の共有prefix計算とgate/up前処理を使い、複数のドラフト候補を形成する。未来planが観測可能でない場合、token-by-token独立損失だけでは系列全体が不整合になることに注意（TM001）。

**対照：** shared MTP head（MA-249）、通常独立MTP head、Medusa/Hydra型の依存ドラフト、Native gain×gain、同byte LoRA、独立joint-plan MTP、必要なら方向間Attentionを比較。MTPM-001では方向間Attentionは事前の追加価値ゲート0/3世界でFAILだったので「方向Attentionを加えれば自動改善」は禁止。

**第一段：** 凍結した共有trunkの最後にMirror SwiGLUを挿入。因果・hidden branch modeで系列NLL、正解系列カバレッジ、target正規化p/q受理トークン数を計測。第二段：小言語モデルの実ドラフト＋ターゲット検証、棄却補正、KV更新、実decode tok/s。受理長だけを速度倍率にしない。

**KV：** 最終Attentionより後の一回のViewでは既存prefix KVを共有しやすい。中間層FFN Viewは後段の隠れ状態・KVを変え得るため、任意の変換でキャッシュを使い回してはいけない（MA-253）。一つのcanonical cacheの厳密なView変換があるならMA-691の条件を別途満たす必要がある。

## 3. 第3候補 P0 — 自然に学習したFFN群の低記述量軌道検査

**目的：** 合成教師は初めからMirror構造なので最重要の外的妥当性が不足している。既存の別タスク／別expert／別レイヤーに含まれる自然な FFN の相互関係がGivens Viewで近似できるかを先に検査する。

**手順：** 実際に別学習されたSwiGLU群から重みを取得。各teacher用に独立したパラメータ最適化を行ったものと、共有物理 \(W_g,W_u,W_d\) にコードを後付けして再現するものを比較。中間チャネルの**同時置換対称性**（gate/upの共通置換とdownの逆置換）を整列させること。ただし任意のゲージ変換はSiLU/要素積を保つとは限らない。教師以外の入力における機能MSE、自然タスクのNLL、rank-1/2 private残差量、実保存byteを必ず確認。

**厳格な境界：** 学習済みのteacherパラメータで共有基底を発見できること（oracle fitting）と、小データから新しいMirrorコードを学習できること（deployment）は別。未学習タスク群は共有基底の学習にもモデル選択にも使用しない。**これがFAILならTransformer全体への大規模投入は延期。**

## 4. 第4候補 P1 — 深さ・周回の機能アドレス（ループTransformer）

**構成：** 周回tごとにgateコード、処理役割rごとにupコードを与えて \(F_{t,r}\) を合成。通常shared recurrent block、per-step scalar gate、rank-1 LoRA、untied blocks、固定depth embeddingを比較（MA-247の開発スクリーニングはFAIL、後の長予算パイロットでは同族教師で逆転例あり）。

**注意：** ループtごとに入力 \(x_t\) が変わるなら \(x_tW_g,x_tW_u\) は**毎周回再計算が必要**。同一入力から複数viewを同時生成する場合の投影共通化を、そのまま反復深さに適用してはいけない。長時間収束、反復残差安定化、周回別勾配干渉の計測が必須。

## 5. 第5候補 P1 — 予算制約付き容量配分（固定自動拡張のFAILを踏まえる）

**観測済みの失敗：** 「訓練済み組の検証損失が停滞したら1→4→12ペアを増やす」単純法は必要な容量を自動最適化できず、0/5で固定12方向との差の品質＋バイト判定に失敗した。これを汎用容量最適化として採用しない。

**新仮説：** 各FFN層lと候補の増分（Givens数、ゲート/Up不均衡、rank-1/2 private、幅追加）について、\(\Delta{\rm validation\ NLL}/\Delta{\rm serialized\ bytes}\) と追加active FLOPs/学習FLOPsを比較し、全体予算下で最も改善が大きい候補を選択する。必要なときだけ private residual を導入する。

**研究規律：** 提案policy、予算、候補、decision gateを新しいdev世界で固定。**fresh task/holdout交差組をcontrollerの選択に用いない。** 単に角度を増やせば良いと仮定しない。各層の簡単な計算/ランダム割り当て/固定大きさとの比較を置く。限られた学習FLOPsを固定し、分散も見る。

## 6. 推奨実行順（簡潔なステージゲート）

| 順 | 実験 | 次に進む条件 | 失敗時の扱い |
|---|---|---|---|
| A | **DMFFN-002** 公平化したSwiGLU gate-only/up-only、gain／FiLM／rank2 LoRA／per-pair Private | 同バイト・同学習計算量で一貫した改善 | 単側コスト修正、原結論は保全 |
| B | **DMFFN-003** 学習済みの自然FFN群が低記述量で共有可能か | heldout自然タスクでNLL/bytesトレードオフが成立 | 大規模Transformer化は延期 |
| C | **DMFFN-004** ゲート×up sparse routing MoEの自然データ | NLL、worst-route、実active FLOPs/byteでnativeを改善 | Mirrorの採用保留、private fallback |
| D | **DMFFN-005** MTP/spec decodeの合成→小型言語モデル | target分布を保つ検証で平均受理数とend-to-end tok/s改善 | 受理数だけなら不採用 |
| E | **DMFFN-006** 周回/層別コードと容量予算controller | 最も簡単なnative制御をbyte/computeでPareto改善 | 自動最適化の主張をしない |

上記 ID はこの**計画書内での作業用ラベル**であり、正式な MA registry ID を予約していない。実装の際は最新 registry と重複検査を行うこと。MAステータスは未変更。

## 7. 先行研究と強対照

- [Masked Gated Linear Unit — NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/dd73f39426a03131c38c8d943153d44b-Paper-Conference.pdf)：一つのSwiGLU内でgate/up行列を無条件に同一化すると能力が崩れ得る。今回の仮説は**各論理MLP間で共通の、ただしgateとupでは別の親行列を共有**しているため、同じ実験とは異なる。
- [2026年ICLRのMoE専門家マージ研究](https://openreview.net/pdf?id=ukGxWd2aDG)：SwiGLU中間チャネルの置換対称性／expert matchingは自然checkpointの比較に必須。
- [ICML 2026 MoE specialization losses](https://proceedings.mlr.press/v306/hu26d.html)：専門家の重複、役割崩壊、ルーティングの偏りを評価するときの強対照候補。
- 既存内製 [MA-251](../../experiments/mirror_applications/ma-251-expert-depth-factorization/README.md), [MA-249](../../experiments/mirror_applications/ma-249-future-head-views/README.md), [MA-247](../../experiments/mirror_applications/ma-247-recursive-depth-view/README.md), [MA-253](../../experiments/mirror_applications/ma-253-cache-safe-final-moe/README.md), [MA-691](../../experiments/mirror_applications/ma-691-lazy-kv-mirror/README.md), [TM001](TM001_PARALLEL_PERIOD_TOKEN_MIXING.md)。

**採用ルール：** 「組み合わせアドレス数」「独立表現容量」「実保存byte」「実active演算」「学習到達までのFLOPs」「自然データ性能」を明確に区別する。合成整合課題の改善のみでTransformer容量の自動最大化が証明されたと言わない。
