# Mirror研究 引き継ぎ — 2026-10-25（5軸スケーリング・Attention・MLP蒸留）

**状態: 研究結果を整理して一時停止。新しい訓練実験・追加ハイパーパラメータ探索は依頼されていない。**

このファイルは `research/ma-1178-pretrained-mlp-mirror-distillation-20261024` の**ブランチ限定ハンドオフ**。過去の失敗結果、固定済み判定、`main` を変更しない。新研究を再開する場合は、まず実験台帳で最新IDを確認し、事前登録と対照を用意すること。

## 1. 最優先の研究目的

単に多数の仮想関数やパラメータを**数え上げる**ことではなく、少ない物理パラメータ、学習FLOPs、実時間、実際のモデル保存量で、通常のDense/FFN/Attention/効率的MoEより有用な機能を獲得すること。

Mirror・Loop・グラフ接続はそれぞれ別の役割を持つ。**Mirrorが最良でないならMirrorを採用しない**というユーザーの方針を優先し、常に強い非Mirror対照を置く。Loopを推論時に残すことも前提としない。

## 2. 5つのスケーリング軸（別々に評価）

| 軸 | 変化させる資源 | 測る量・対照 | まだ証明されていないこと |
|---|---|---|---|
| Loop | 逐次状態更新の回数、学習中の教師計算 | 固定パラメータでT=1/2/3、通常の1回計算、蒸留後の推論FLOPs | 多段計算が1回へ自動的に定着すること |
| Mirror | 共有物理核から作る関数の違い、変換方向・コード | 同係数・同実行予算のFiLM、低rank、shared、independent、関数空間誤差 | 仮想状態の数だけ独立自由度が増えること |
| 物理パラメータ | 実際の学習可能係数、保存バイト数 | width/係数数を変えるPareto frontier | 係数を増やしただけの品質改善がMirror固有であること |
| データ | ユニーク事例、分布・品質・ルーティング | データ量とデータ品質を別々に調整、希少規則・OOD評価 | 学習に使う量が多いほど必ず汎化すること |
| 探索 | 候補提案・選択・将来評価の追加計算 | 同じ**総学習FLOPs**を追加学習へ回す対照、no-change shadow | 候補数を増やせば計算効率が良くなること |

**Loopは学習時の計算を推論時1回へ償却／蒸留できるかという仮説**、Mirrorは機能的な共有と配置最適化、探索は実際に学習費用を回収できるかを問う。これらをまとめて「スケールが大きい」と言わない。

## 3. Attentionとの接続は仮説、同一視は禁物

MiTA ([arXiv:2602.01219](https://arxiv.org/abs/2602.01219)) はAttentionを**文脈依存の2層Fast-Weight MLP**として扱う。これは「入力と無関係な固定重みMLPが一般のAttentionに等価」という主張ではない。softmaxは隠れユニット間で正規化を共有する。

有望な追加仮説は、Q側のみ小さなMirror/Viewを適用し、K/Vを共有した**部分Mirror化された仮想Attention head**を、入力依存で少数だけ実行すること。QとKに同じ直交変換を施すと内積が不変になり、単にビュー数を増やしても機能は増えない。Q側Mirrorだけでは参照するK/Vの数は減らない。学習効率を目的とするなら、MoH/MoBA/MiTA・効率的なカーネル・ルータ費用に対する公平な比較が必要。画像用MiTAの`causal=False`を自己回帰LMへそのまま流用しない。

非Mirrorの通常head、FiLM/low-rank、独立expert、dense Attention、KV疎化との比較を、**真のAttentionスコア、因果性、保存量、学習時間**で行う。手元の36-model Q-Mirror人工教師probeは探索的診断であり、自然言語や学習速度の証拠ではない。

## 4. 完了済みの重要実験（判定を書き換えない）

| ID | 事実・主要な結果 | 科学的な境界 |
|---|---|---|
| [MA-1173](../../experiments/mirror_applications/ma-1173-lookahead-supergraph-lm/REPORT_JA.md) | 54 tiny symbolic causal LM比較。未来を見た経路切断は改善0/6 | 実際の自然言語ではなく、構造選択の有効性はFAIL |
| [MA-1174](../../experiments/mirror_applications/ma-1174-structure-guided-supergraph/REPORT_JA.md) | 66合成回帰モデル。切断で共有核勾配の方向は変化するが性能改善0/6 | 当時の候補選択は「切らない」選択肢を持たず強制切断 |
| [MA-1175](../../experiments/mirror_applications/ma-1175-corrected-gradient-intervention/REPORT_JA.md) | 100モデル。経路別共有核VJP合計誤差~1.67e-7、no-cut guard 20/20、学習品質のゲート0/10 | 機構はPASS、構造介入の改善はFAIL。追加学習が優勢 |
| [MA-1176](../../experiments/mirror_applications/ma-1176-five-axis-scaling/REPORT_JA.md) | 189極小モデル。Loop T3対T1は3/3改善、1回蒸留1/3 FAIL、Mirror対同係数FiLM 2/6 FAIL、探索4候補対追加学習3/6 FAIL | Loopの改善が即「推論1回への蒸留成功」を意味しない |
| [MA-1177](../../experiments/mirror_applications/ma-1177-softmaxfix-mirror-attention/REPORT_JA.md) | 120モデル。安定CEのMirror汎化改善は0/6、完全一致教師への蒸留ドリフト解消6/6 | Barronの安定化は汎用数値改善。FiLMにも効き、Mirror固有性能ではない |
| [MA-1178](../../experiments/mirror_applications/ma-1178-mlp-mirror-distillation/REPORT_JA.md) | 20学習済み通常MLP教師、180学生。構成上整合する3教師の444係数を196係数Mirrorでoracleほぼ完全再現（DEV NMSE最大4.72e-14） | **表現可能≠ランダム初期値から学習可能。** 主H16の関数再現<.01は2/5、全世界Mirror対FiLM改善3/15、主ゲートFAIL |

MA-1178の重要な追加観測：Mirror整合教師H8で、100係数のMirrorはFiLM100係数・独立93係数よりFresh NMSEを10%以上改善したペアが5/5。ただし教師がMirror形式で構成された有利な条件。H16ではMirror 0.010794 / FiLM 0.012171、独立教師では通常独立MLPの方が良い。CPU学習・推論時間では同係数FiLMよりMirrorが遅かった。事後の**事前学習済み共通核転移**40モデルは探索的で、確認済みの一般的優位性に昇格しない。

各IDの`PROTOCOL.json`、`STATUS.md`、`results/VERIFICATION.json`、`RESULTS_CORE.csv`と研究ブランチを一次証拠とする。large checkpointはGitHubに直接入っていない場合があり、ハッシュや成果物パッケージの存在を実ファイルと混同しない。

## 5. 現時点の設計原則（結果ではない）

1. **機能を教師から直接蒸留して診断**：重み差でなく出力関数を比較。教師がMirror整合、FiLM整合、独立の各分布で結果を分ける。単一関数なら単純なDenseに吸収できる場合があるので、複数関数共有の条件で価値を測る。
2. **Mirrorは必要な場所にだけ挿入**：全層Mirror化しない。最初にどのQ/K/V・FFN内部で独立した機能変化が生じるかを確認。
3. **Loopは推論時に消える可能性をテスト**：学習時反復の終端関数を、一回実行の学生・静的グラフへ蒸留して評価。計算を隠さない。
4. **グラフ構造とデータルーティングを混同しない**：候補が多くても実行する分だけが実計算。接続選択はno-cut、初期から同じ最終トポロジー、独立expertを必ず比較。
5. **自然言語での効率改善は依然として未実証**：Attentionを学習済みモデルから保持したFFN置換、または入力依存Fast-Weight教師を検討する場合、自然文NLLと真のtrain-FLOPs-to-target、CPU/GPU学習時間を厳密に比較する。
6. **Softmax数値安定性**：高確信度CEと近一致蒸留では [Barron et al., 2026](https://arxiv.org/abs/2610.31642) の問題を監査。これは機構の数値的正しさであり、Mirrorの容量増加の証拠ではない。

## 6. 次の作業に関する停止線

**2026-10-25の指示は「とりあえずリポジトリ更新」。追加実験をこの指示から推定して実行しない。** MA-1178はすでに終了した研究として保存済みで、再学習・条件変更・結果の上書きをしない。次の実験の優先順位は再開指示が出た時点で決める。

復帰時の手順：`AGENTS.md` → `WORKER_START_HERE.md` → この文書 → 直近MA報告書と`experiments/mirror_applications/EXPERIMENT_CONTRACT.md`。最新`IDEA_REGISTRY.csv`から重複しない次IDを割り当て、README・PROTOCOL・STATUSを**学習前に**コミットし、Freshを最後まで封印する。

**失敗は失敗のまま残す。** 全面採用・独立Expertと同等・無料の演算・学習効率改善・普遍的容量倍率を、今回の証拠から宣言しない。
