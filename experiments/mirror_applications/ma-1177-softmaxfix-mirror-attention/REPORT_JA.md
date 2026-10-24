# MA-1177 — Cross-Entropyの数値修正はMirror Attentionを改善するか

**実施日：2026-10-24（JST）**  
**結論：数値機構PASS／Mirrorの通常分類品質の改善FAIL／同一関数蒸留の安定化PASS／Mirror固有の優位性FAIL。**

## 1. 研究上の位置付け

Barron, Gnanapragasam & Tschannen (2026), [*A Fix for Softmax and Cross-Entropy*](https://arxiv.org/abs/2610.31642) は、(A) 高確信度の正解クラスの微小勾配が通常CEでは失われる **Softmax Collapse** と、(B) Student/Teacherが近い時に確率を別々に求めて引くことで不正な差分が発生する **Softmax Cancellation** を指摘する。原論文は https://jonbarron.info/softmaxfix/ 。

**問い：**これらの数値誤差は、少数の共有物理係数から条件付きに仮想Attention Expertを作るMirrorの学習・機能保存に影響するか。また、その改善は単純な同係数数のFiLMよりMirror特有のものか。

前回の通常マージンでのMirror Attention教師KL監査は変化が最大約 `1.2e-7` にとどまった。本実験は**高確信度のhard-label分類**と**Teacher完全一致からの蒸留**へ意図的に対象を広げた独立の `MA-1177`。過去のMA-1173/1175/1176のFAIL判定を上書きしない。今回の実験は**Grokkingや自然言語事前学習の実験ではない**。

## 2. 事前登録・条件

訓練前に `README.md`, `PROTOCOL.json`, `STATUS.md` を `research/ma-1177-softmaxfix-mirror-attention-20261024` ブランチへコミット。プロトコルSHA-256：`60277fbc83524c49c44cb1c825a8a1d599a78355b57d7ab97ef80d7a34cd8e08`。

- **入力**：8次元クエリ、12件の過去メモリ、3規則。ルールIDは入力として提供される**oracle routing**。未来情報は使わないが、フル自己回帰LMではない。
- **教師世界**：Householder整合、規則ごとに独立した直交写像。ランダムシード 1801, 1811, 1823、train 384 / dev 256 / fresh 768 / OOD 768例（各split独立）。OODは入力の振幅を変化。
- **物理係数**：Mirrorは共有8×8写像＋3×8 Householder軸で88係数、FiLMは同じ共有8×8＋3×8ゲインで**88係数**、独立Expertは3×8×8で192係数。学習可能なRouterはなし。
- **分類実験**：2世界×3シード×3方式×スコア倍率2.5/12×従来CE/修正CE＝**72モデル**。各550 AdamW更新、バッチ64、LR0.016、eps1e-8、weight decay0.05、FP32 CPU 1スレッド。同seedでは同じデータとバッチ順序。
- **蒸留**：分類を修正CE・倍率12で学習したMirror/FiLM教師を凍結。Studentを教師**完全一致コピー**から開始。2世界×3シード×2方式×eps1e-8/1e-12×従来soft CE/修正soft CE＝**48モデル**。150更新、LR0.02、weight decay0。これは近似能力ではなく、**動く必要のない重みが移動するか**の検査。
- **開封順序**：72＋48モデルすべてのチェックポイントとSHA-256を `results/SEALED_TRAIN.json` に固定後、初めてFresh/OODを生成。Freshに基づくパラメータ選択なし。

**計算量と保存量の境界**：同じパラメータ化内で従来/修正損失は**同じ学習済み係数数・主更新数**だが、修正版は別の数値演算を加える。独立Expertの保存量はMirror/FiLMより大きい。PyTorchの実測学習時間を別に記録。チェックポイントの`.pt`サイズにはメタデータが含まれ、推論用の最小保存量とは区別する。

## 3. 数値誤差は再現できる（機構PASS）

| 数値形式／例 | 通常CE：正解logit勾配 | 修正CE：正解logit勾配 | 修正の勾配和 |
|---|---:|---:|---:|
| FP32：`[18,0,0]` | 0 | **−3.046×10⁻⁸** | 0 |
| BF16：`[7,0,0]` | 0 | **−1.823×10⁻³** | 0 |
| FP64：`[40,0,0]` | 0 | **−8.497×10⁻¹⁸** | 0 |

実際に学習したMirror（Householder整合・倍率12）のtrain384例でも、FP32のmargin16.6を超えたものは**5／1／13件（3シード）**存在した。その代表例の通常CE正解側勾配は0、修正CEは非ゼロとなり、FP64参照勾配と一致した（例えばseed1801：FP64 `−1.42545353e−9`、修正FP32 `−1.42545398e−9`）。また同じ問題はStandalone softmaxの最大確率勾配でも生じた。

注意：修正版のAttention softmaxをFP64の小fixtureで検証したが、今回の学習本体では**FlashAttentionやPyTorch SDPA内部のSoftmaxカーネルを置き換えていない**。実装したのは分類CEと蒸留soft-target CEである。

## 4. 高スコアの分類学習：最終品質はほぼ変わらない

**3シード平均Fresh FP64 CE（nat/問い合わせ、小さいほどよい）、スコア倍率12**。

| 教師世界 | 方式 | 従来CE | 修正CE | 学習中にmargin>16.6のtrain例割合（従来） |
|---|---|---:|---:|---:|
| Householder整合 | **Mirror 88** | 0.488817771 | 0.488817773 | 1.649% |
| Householder整合 | FiLM 88 | 1.39209 | 1.39209 | 0% |
| Householder整合 | 独立Expert 192 | 0.67242 | 0.67242 | 4.427% |
| 独立直交教師 | Mirror 88 | 1.88864 | 1.88864 | 0% |
| 独立直交教師 | FiLM 88 | 1.97362 | 1.97362 | 0% |
| 独立直交教師 | 独立Expert 192 | 0.64390 | 0.64390 | 3.646% |

**72モデル・全36ペア**における従来/修正CEのFresh差の最大絶対値は `2.7891e−7` nat/問い合わせ、最大相対変化は `4.7143e−7`（相対比）だった。倍率12のMirror6条件で事前閾値の5%以上Fresh改善は **0/6（FAIL）**。FiLMに対する「修正による改善率が5パーセンテージポイント以上大きい」条件も **0/6（FAIL）**。

**なぜ改善しないのか（解釈）**：高マージンで消失する勾配は微小で、そもそも強い性能差の主要因にはならない可能性がある。今回のタスクは大量に暗記したアルゴリズム的なGrokking課題でなく、飽和領域にいる正解例が限定的。したがって原論文のGrokking効果を反証する結果ではない。

また、Householder整合教師でMirrorがFiLMより強いのは**教師構造への整合**によるもので、Softmax修正による性能改善ではない。独立直交教師では、独立Expertが両共有方式に大きく勝つ。

## 5. 完全一致蒸留：Mirrorの機能保持は大幅に変わる

**教師と完全同じ重みから開始したStudent**に、正しくはゼロである学習更新を150回実行した。FP32・AdamW eps1e-8・weight decay0。以下は3シード平均のFresh Teacher–Student KL（nat/問い合わせ）。

| 教師世界 | 方式 | 従来Soft CE | 修正Soft CE |
|---|---|---:|---:|
| Householder整合 | Mirror 88 | **0.027916** | **0.000000** |
| 独立直交教師 | Mirror 88 | **0.009072** | **0.000000** |
| Householder整合 | FiLM 88 | 0.003167 | 0.000000 |
| 独立直交教師 | FiLM 88 | 0.000829 | 0.000000 |

- Mirror eps1e-8の6/6世界・シードで修正版は教師へのKLが**厳密に0**、全パラメータの移動も0。一方、従来版は6/6でTeacherと異なる関数へドリフト。
- Mirror eps1e-8の従来版の6ペア平均KL：**0.018494**、中央値約0.015011、Freshで教師のTop-1と異なる割合平均**5.92%**。修正版はTop-1不一致率0%（全く同じ出力）。事前ゲートは**6/6 PASS**。
- 従来版では入力logitへの誤差勾配が完全一致時でも各条件およそ `1e−8` 程度あった。修正版は厳密なゼロ勾配を返す。Adamの小さなepsとの組合せでドリフトが増幅された。
- FiLMでも同じ現象が発生するため、**数値修正の効果はMirror固有ではない**。ただし今回の条件では従来版MirrorのKLドリフトの方がFiLMより大きかった。モデルごとの感度差をMirror一般の性質とみなすことはできない。
- eps1e-12でも、Mirror/FiLMに対し修正版の機能維持が確認された。

**重要な限界：**教師と生徒が完全同一なら、教師の分布を維持するだけで「性能改善」に見えるのは数値的恒等性の検査である。**別の構造・少ない係数のStudentが教師をよく近似できるか、という圧縮蒸留を示したものではない。**

## 6. 事後探索：少しだけ異なるStudentでの転移（DEVのみ）

確証実験のFresh開封後に追加したため、`exploratory/NEAR_TEACHER_DEV_ONLY.csv` は**事後・DEV限定の診断**として区別する。教師から初期重みをノイズσ=1e−4または1e−2だけずらし、LR0.02・eps1e−8の150更新をした（2世界×3シード×2方式×2損失×2σ=48条件）。Freshは使用しない。

| Mirror Studentの初期重みノイズ | 初期DEV KL | 従来Soft CE 150更新後 | 修正版Soft CE 150更新後 |
|---:|---:|---:|---:|
| 1e−4 | 0.00000205 | 0.0141746 | 0.0141730 |
| 1e−2 | 0.0184972 | 0.0117085 | 0.0117085 |

**機能がわずかでも異なると、修正したCEだけでは不適切な最適化更新を防げない。** ごく小さい初期誤差に対してLR0.02のAdamW更新が過大となる可能性が高い。この結果は蒸留時の学習率・勾配制限・局所曲率も別に調べる必要があることを示す。事後試験なので、確認的な改善主張には使わない。

## 7. 実行コストと誤差

同じモデル・更新数における本Python CPU eager実装の、550回hard学習1モデル平均壁時計は通常CE約**0.258秒**、修正CE約**0.307秒**。150回蒸留は通常Soft CE約**0.093秒**、修正Soft CE約**0.124秒**。いずれも今回のCPU環境・Python実装に固有で、最適化済みGPUカーネルへの外挿はしない。学習時間改善は立証されず、今回はむしろ修正版に追加オーバーヘッドがある。

蒸留eps1e−8におけるMirrorのペア別KL改善量の標準偏差は約0.01745、平均差の記述的合成標準不確かさ `u_c≈0.00712`（k=2なら幅約0.01425）。これは教師世界2種類×各3seedの混合集計であり、集団母数の信頼区間と扱わない。完全一致という機構検査の有限精度誤差は別に扱う。

## 8. 判定（事前ゲート変更なし）

| 主張 | 判定 |
|---|---|
| FP32/BF16等で損失・softmax勾配の数値機構が正しくなる | **PASS** |
| 修正CEによりMirror高margin hard学習のFresh CEが5%以上改善（6条件中4以上） | **0/6 FAIL** |
| 修正Soft CEで完全一致MirrorのTeacher–Student機能保持が改善（6中5以上） | **6/6 PASS** |
| 上記分類効果が同数係数FiLMよりMirror固有に大きい | **0/6 FAIL** |
| 少ない計算・少ない保存量で汎用のMirror Attention優位を示す | **FAIL / 未立証** |

### 原因と次の実験

- 「数値問題が確かにある」ことと「当該モデルの汎化性能が改善した」ことを分ける。CEだけ修正しても、本モデルの分類品質はほぼ変わらなかった。
- **Multi-Loop教師→1回実行Mirror Student** の蒸留や、文脈ごとのMirrorアドレス転移には、この修正Soft CEを安全な数値基盤として取り入れる価値がある。通常の蒸留初期化との比較、FiLM・低ランクとの同係数予算比較が必須。
- **真のGrokking検証**には、記憶後にも学習を継続する離散演算課題、margin・Gradient zero-sum・Adam eps・weight decay・旧能力保持を制御した長期学習が別途必要。今回これらを学習曲線で検証したとは主張しない。
- Attention内部のsoftmaxを差し替えるときは、PyTorch SDPA/FlashAttentionの実装を直接変更する必要がある。今回の実験からGPU/FlashAttentionの高速化を推定しない。

### H/T/D/C/U

- **H**：数値修正がMirrorの高確信度分類学習・教師一致蒸留を改善し、独立したMirror特有の優位性を生むか。
- **T**：同一学習係数・シード・訓練順序で標準/修正hard CE・soft CEを比較、FiLM・独立Expertを対照、Freshは全checkpointを固定後に開封。
- **D**：数値機構PASS、厳密一致蒸留PASS、通常分類品質0/6 FAIL、Mirror固有優位0/6 FAIL。
- **C**：改良効果は数値安定化という一般問題であり、Mirror表現自体には付加的な表現自由度を生まない可能性。完全一致なら固定点保持は当然であり、実用的な圧縮転移とは別。
- **U**：各世界3seed・人工データ・oracle rule・FP32 CPU・極端なスコア倍率。真のGrokking、GPUカーネル、自然言語、各深さの反復蒸留への外挿には追加検証が必要。

## 9. 再現証拠

- `PROTOCOL.json`：訓練前にGitHub commitで凍結。SHA-256 `60277fbc83524c49c44cb1c825a8a1d599a78355b57d7ab97ef80d7a34cd8e08`
- `source/core.py`, `source/run.py`, `tests/test_numeric.py`：**8/8 tests PASS**。
- `results/SEALED_TRAIN.json`：**120件checkpoint SHA256の封印**、Fresh未生成フラグあり。ファイルSHA-256 `563f2125354d4bdd4d1c328964e4baa5d1760fb1b15da6de58939ddc2b458b27`
- `results/RESULTS_CORE.csv`：全120件のFresh/OOD、SHA-256 `a1b66724699b22ce25e2a07d0bc3b41e56fb0b59a3c178ea98d309451ac0bb02`。
- `results/ANALYSIS.json`, `results/NUMERIC_AUDIT.json`, `exploratory/NEAR_TEACHER_DEV_ONLY.csv`。
- `source/verify.py` による**120/120 checkpoint replay一致、Fresh値一致、hard 1件とdistill 1件の重み完全再訓練一致PASS**。`results/VERIFICATION.json`に証拠。
- `checkpoints/`：全120チェックポイント（配布ZIPに同梱）。通常GitHubソース履歴には軽量コード・レポートとCSV/マニフェストを保存し、全チェックポイントはZIPで取得可能にする。
- PyTorch 2.10.0+cpu, Python 3.13, CPU 1-thread, FP32、時計固定なし。

**科学的な判断：** Softmax修正は、共有物理係数で作るMirror仮想Attentionの**蒸留・機能保存を数値的に信頼できる状態にする重要な修正**だが、Mirror固有の学習効率や容量を増やすという主張ではない。
