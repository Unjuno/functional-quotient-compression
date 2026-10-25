# Mirror 研究追補 — 2026-10-25：関連教師蒸留とoptimizer幾何

**ブランチ限定の引き継ぎ。** `main` とMA-1178以前の既報を変更しない。

## 背景と再開経緯

以前の [MLP・スケーリング・Attention 引き継ぎ](RESEARCH_HANDOFF_2026-10-25_MIRROR_SCALING_AND_DISTILLATION_JA.md) は、その時点の「文書更新だけ」という指示に対応していた。以後、ユーザーが明示的に「実験を進めとって」と依頼したため、新しい事前登録済みMA-1179を作成した。さらに「Mirror自体がoptimizerと相性が悪い可能性」を指摘されたため、MA-1180を**別仮説・別ID**として事前登録した。

## 実験済み事実

| ID | 概要 | Fresh事前登録合否 |
|---|---|---|
| [MA-1179](../../experiments/mirror_applications/ma-1179-correlated-teacher-core-warmstart/REPORT_JA.md) | 35教師・120学生、機能だけからの共有核250更新ウォームスタート vs コールド、独立学習した関連教師を含む | Mirror warm改善1/5 FAIL、FiLM/rank1双方へのMirror固有優位0/5 FAIL |
| [MA-1180](../../experiments/mirror_applications/ma-1180-mirror-optimizer-geometry/REPORT_JA.md) | 同じMA-1179教師SHAを使用、新しい学習/DEV/Fresh/OOD乱数列、120学生、8 optimizer/方式。Mirror Householderコード方向の球面更新を検証 | 通常AdamW比10%以上改善3/5 FAIL、DEV選択したAdamW4候補比0/5 FAIL、FiLM比0/5 FAIL |

**機構と効用を混同しない:** MA-1180ではFP64のスケール/符号不変性と反射コード勾配の接空間性を検算。AdamWの移動には半径方向余分成分があり、再正規化/球面更新で抑制できた。一方、関連教師Fresh NMSEはAdamW Mirror中央値0.020892、球面スカラー適応0.021933、FiLM0.012200。球面修正は有効な**幾何的補正**だが、観測条件での品質優位にはならなかった。

## 検証・コード

- MA-1180: 全120チェックポイントSHA256とFresh出力の再生120/120一致、教師SHA256 5/5一致、数理・source機構テスト10/10 PASS。
- `PROTOCOL.json` → `source/run.py` → `results/SEALED_TRAIN_MANIFEST.json` → `results/FULL_RESULTS.json` → `results/ANALYSIS.json` → `results/VERIFICATION.json` の順に確認。Freshは学習完了後の封印より後だけ。
- 両実験はFP32 CPU一スレッドの合成4D teacher蒸留。**自然言語LM、GPU、強い一般化容量倍率を意味しない。**
- MA-1180の教師バイナリはMA-1179から再利用。巨大バイナリ自体がGit履歴にあると誤認しない。リポジトリにはSHA256・生成コード・結果テーブルを収録。

## 次に検証すべき仮説

最も重要な未解決問題は「実際の関連学習済み関数群の差を、Householderの低次元Viewで表現・学習できるのか」であり、`FiLM` / 同費用低ランク / shared＋private補正と同時比較が必要。Mirrorコードだけのlr最適化に無期限に計算を投じない。さらに自然言語FFN/Attentionへの転用では、NLL・train FLOPs-to-target・保存bytes・CPU/GPU時間のParetoと、教師系統外OODを要求する。

新ID割り当ては現ブランチの`experiments/mirror_applications/IDEA_REGISTRY.csv`を再読し最大ID+1を利用。過去のFAILを上書きしない。主研究方針は依然として「Mirrorが有効でなければ採用しない」。

