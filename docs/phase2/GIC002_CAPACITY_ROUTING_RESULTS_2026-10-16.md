# GIC002 — 変形自由度・初期配分・関数保存拡張

2026-10-16、コンテナCPUで実行・同一セッション検証済み。既存研究結果や正式MA判定を変更しない。

[日本語結果](../../experiments/mirror_applications/research_intake/gic002_capacity_routing_20261016/SUMMARY_JA.md) / [コードと再現方法](../../experiments/mirror_applications/research_intake/gic002_capacity_routing_20261016/README.md) / [数学的契約](../../experiments/mirror_applications/research_intake/gic002_capacity_routing_20261016/MODEL.md)

- 2専門家のまま1→8方向を増やすと、4区分の品質ゲートが通過。シード絶対分散＋品質の同時条件は1区分のみ。
- 初期だけ50/50は同族512例の副次比較で改善。125更新の強制均等warmupは一般的な改善なし。
- 新しい5世界における関数保存の途中拡張は、狭い対照に両族5/5で改善。最初から同容量の対照への優位は限定的。
- 独立生成教師では同容量native gainが強く、Mirror実行時間は不利。初期ロックインや過去失敗の原因を確定しない。
- 主評価2240モデル、校正224、拡張320＋親80、事後参照160。19テスト、5808保存ファイル、40完全再学習照合。

親記録: GIC001 at a625fd90f2b8827ff33faa992b0498419f577626。今回の小型GELU/MoE実験はその固定曲線実験やAMA-001の再実行ではない。
