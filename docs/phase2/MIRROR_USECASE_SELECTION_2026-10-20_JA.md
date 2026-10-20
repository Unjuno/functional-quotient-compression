# 2026-10-20 — Mirror × Loop の用途を絞り込む（研究ブランチ限定）

**状態：研究方針の更新。実験を新たに実行したという意味ではない。** 凍結済み MA-1163/1164/1165 の否定的結果や MA-1166 Phase A–E の評価を変更しない。main へはマージしない。

## 結論

Mirror を「仮想重みの数を増やせるので表現自由度が増える」仕組みとして売り込むのは不適切。**少数の物理学習係数を多数の仮想重み・仮想ノード・非隣接接続・Loop状態に対応付け、その *対応付け自体* を学習して、最後に小さくコンパイルする学習用構造探索器**として検証する。

**優先探索対象：trainable many-to-many physical→virtual address mapping。** 現状のデコーダは主に固定シードのアドレスを使っており、この中心仮説は未検証。

## 現在の証拠：混同しない

| Evidence | Actual scope | Verdict | Not established |
|---|---|---|---|
| MA-1163/1164 | 学習した単一ベクトルから数字分類MLP重みへ非線形Mirrorデコード | Mirror以外の同程度の重み表現と比較して固有利得なし。コード・実バイト費用も課題 | 圧縮された任意ネットワークを任意アドレスから復元できる |
| [MA-1165](../../experiments/mirror_applications/ma-1165-step-switched-virtual-mlp/REPORT_JA.md) | 192係数の一物理テンソル、T2/T4、72 CPU runs | step-switched nonlinear Mirrorは実関数を変えるが、独立教師T4の簡単な段別ゲインを超える条件1/3。**FAIL** | Mirror別の意味を最適に配置できる |
| [MA-1166 Phase C audit](../../experiments/mirror_applications/ma-1166-exhaustive-mirror-supernet/PHASE_C_AUDIT_20261020.md) | 253係数のうち共有核64、63経路、固定アドレス | aligned教師はoracle経路指定でほぼ再現できるが、逆伝播は経路指定を発見しなかった | 未知構造の発見・自然タスクへの圧縮。監査のソース/チェックポイントはGitHub単独では再実行不可 |
| [MA-1166 Phase E](../../experiments/mirror_applications/ma-1166-exhaustive-mirror-supernet/PHASE_E_REPORT_JA.md) | 442係数=共有核64+126 gate+252 view codes、126仮想MLP×110重み、36 synthetic CPU runs、全辺非剪定 | SYNC3対SERIAL Mirrorはfresh 5/6改善するが **378対126辺実行**。affine SYNC3を超えるのは1/6、native442は0/6。8/8機構テスト、36/36再評価は同報告に記録 | 13,860個の独立自由度、並列化によるGPU高速化、Mirrorの圧縮優位。元チェックポイント・ソースのGitHub全格納も未了 |
| [InfiLoop](../../experiments/mirror_applications/research_intake/infiloop_loop_native_residual_20261020/README.md) | 外部研究、Loop状態の content-weighted EMA | Latent overthinking対策の先行実証 | Mirrorのアドレス学習・物理刈り取りの実証ではない |
| [Virtual Head Attention](https://arxiv.org/abs/2610.17959) | 外部研究、線形head Premix/Postmixの学習と推論畳込 | 変換の**学習上のパラメータ化**に利点があり得る | 任意の仮想非線形FFN/経路を無コストで追加できること |

**「1パラメータ」はあくまで1個の学習テンソルのこと。** MA-1166 Phase E には442個の実スカラー係数がある。共有核は64のみであり、gate+codeの378個（85.5%）は経路固有の学習係数である。生の仮想重み数13,860を物理学習自由度と呼ばない。滑らかな生成器の仮想重みJacobian階数は高々物理学習係数数。

## 用途の優先順位：候補は4＋保留2に絞る

### P0 / 実験一番：物理↔仮想パラメータの対応付けを学習する

- **利用場面:** 最初に大きな仮想スーパーグラフを保持し、学習中にどの物理係数/共通機能がどの仮想ノード/辺で意味を持つかを決める。最終モデルを少ない実バイトへコンパイル。
- **Mirror insertion:** `w_e = D(u, m_e, A_e)` の `A_e` を固定乱数アドレスから低記述長の**学習可能な読み出し/配置**へ変更。既存の **weight-space Q→tanh→Q^{-1}** デコーダを残し、安価な線形/アフィン版も直接比較する。
- **勝ち筋の条件:** alignedとunseen独立教師でマッピング/結線が発見でき、同予算の学習可能な線形割当てやFiLM/LoRAおよび通常MLPと同等以上の精度–総バイト–active MACを実現すること。
- **第一停止条件:** oracle-known edge IDs だけ与えても（重みやMirrorコードを教師からコピーせず）学習が失敗する、または学習コードの保存量が独立重みのバイトを超えるなら、探索規模を増やさない。
- **次の文書:** [Phase F 提案](../../experiments/mirror_applications/ma-1166-exhaustive-mirror-supernet/PHASE_F_TRAINABLE_MAPPING_PROPOSAL.md)（**未実行**）。

### P1 / 次点：学習用の巨大グラフから推論用の小さな固定グラフを蒸留する

- **利用場面:** 全ノード間（隣接/飛越/再訪）と仮想重みへの割当てを連続ゲートで学習。学習途中のhard pruneはせず、学習終盤に二値化し、出力パッチ/再学習で確認し最終的に刈り取り。
- **現状の負の証拠:** MA-1166 Phase C/Bの素朴なゲートでは支持経路が自然には十分消えず、閾値のみでは誤判定。Phase Eは全126候補を最終推論まで実行しているため推論圧縮ではない。
- **判定:** 検証タスク損失への削除影響・少額再最適化後の復旧・共同削除の相互作用を同時に計測。共有物理係数を削除するには残存すべての仮想用途からの参照消失が必要。
- **出力:** (a) 展開済み疎な標準モデル、(b) 共有基盤を残した圧縮モデル、双方の実バイト・総MAC・壁時計を測る。

### P1 / 条件付き：高価な共有計算を1回だけ行う複数出力

- **利用場面:** 高価な共通射影 `z=Wx` を1回計算して、複数の独立した軽量Mirror読み出し・後続活性化・選択機能を得る。
- **境界:** `P_k z` は軽量にできても、多数の独立した大規模FFNや非線形Attention scoreを計算しないで済む証明にはならない。直列Loopの `h_{t+1}=F(h_t)` を1回の独立読み出しで厳密代用できるとは限らない。
- **比較:** matched-price FiLM、shared projection + diagonal/scalar gate、low-rank, explicit multiheads。推論で `P_kW` へ畳み込んで重み複製する場合の保存量も計上。

### P2 / 付加機構：InfiLoop式のLoop出力保護

- **利用場面:** Loopが過去の正しい状態を上書きする synthetic reasoning, iterative repair で、Mirror候補からの出力をcontent-weighted running residualで受け入れる。
- **境界:** 受入係数 `gamma_t` は「候補状態を一時的に無視」する度合いであり、仮想ノードや物理係数が不要という**剪定シグナルではない**。
- **比較:** carry-last / 固定EMA / 学習scalar damping / InfiLoop型 / 各々Mirrorとaffine。追加q/beta状態・BPTT費用を計上。

### HOLD / KV cache Mirror使い回し（独立に証明してから）

- **推論前条件:** カノニカルなK/Vを低価格で保持し、逆Mirrorで他Viewへ変換できる場合のみ。QとKを同じ直交Rで変換すれば内積不変なので**その部分だけでは新しいAttention機能が発生しない（gauge）**。異なるrouting/softmax分布の追加計算を無料とは呼ばない。
- **必要監査:** causal mask・softmax score・value/readout・recompression、再投影とKVメモリ比較。性能と実行費用でGQA/MLA等と比較。

### HOLD / Loop増加そのもの・Grokking・Mirrorデコーダ単独

- **理由:** 現在のMA-1165/1166では単純gain/affine/MLP対照に品質・計算の劣位がある。多Loopを実行しただけで表現自由度やGrokkingが増加するとは断定できない。
- **重要:** GrokkingのPhase Dは[事前プロトコル](../../experiments/mirror_applications/ma-1166-exhaustive-mirror-supernet/PHASE_D_GROKKING_PROTOCOL.md)が存在するが、GitHubブランチ上に検証可能な結果表・重みはない。未検証として扱う。

## 数学的な設計原則

1. 物理係数 `u∈R^P` と仮想重み `w_e∈R^N` を区別。接続 e のコード・共有基底・住所付け `m_e,A_e` の**実保存量**を計上。
2. 可逆な直交反射 `Q^{-1}=Q^T` と、非可逆な `tanh/ReLU` を混同しない。Mirrorが可逆だからFFN変換が一般に逆に戻せるという意味ではない。
3. 多対多 `e=(source,target,loop,mirror)` を許す。ループ展開DAGの同期Jacobi更新は同一反復内で並列計算できるが、時間依存する反復間は直列障壁が残る。
4. 学習可視のスコアで自然に経路が消えるとは仮定しない。まずoracle route supportだけ知らされた場合（oracle weight/codeは使わない）から正しく学べるかを検証し、次にsupport未知へ進む。
5. 初期全候補の完全部分グラフとランダム部分計算を**別の学習目的関数**と認識し、fullに対するunbiased gradientを証明していないなら同値とは呼ばない。
6. 最優先性能指標は **未使用タスク品質 / 総実保存B / 実行MAC / wall time / compile後品質** のParetoである。学習時の論理重み数や候補数は容量証拠にならない。

## 更新範囲と実験状態

- この文書の追加は**研究仮説・優先順位の更新**。新たな重み学習、GPU測定、言語モデル評価、試験結果変更は実施していない。
- MA-1166 は開発段階 SCREENING（Phase Eの数値報告はあるが GitHub単独の完全再現パッケージではない）。新しいPhase Fは**PROPOSED / NOT RUN**。
- 凍結済み否定的判定は一切変更しない。将来の採否では MA-1166 と通常/affine/FiLM/low-rankの固定物理係数・実B・MAC対照を必須とする。
