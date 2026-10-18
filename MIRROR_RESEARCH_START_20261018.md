# Mirror共有核研究 — このブランチの入口

**2026-10-18 / 整理のみ・新規実験結果なし**

研究ブランチ：`research/mirror-shared-core-agenda-20261018`

基点：`747eb68d63c7ac1c97aece06227658a8d65f04cf`（`research/ma1162-multihead-mirror-fold-20261018`）。mainや既存研究の状態は変更しない。

## 読むもの

1. [目的・用語・スケーリング軸・証拠の整理](docs/phase2/MIRROR_SHARED_CORE_RESEARCH_CHARTER_2026-10-18.md)
2. [コンテナで進める作業順序・対照・合否条件](docs/phase2/MIRROR_SHARED_CORE_EXECUTION_QUEUE_2026-10-18.md)
3. 既存の[AGENTS.md](AGENTS.md)、[WORKER_START_HERE.md](WORKER_START_HERE.md)、[研究ドクトリン](docs/phase2/MIRROR_PARAMETER_INTEGRATION_DOCTRINE.md)、[実験契約](experiments/mirror_applications/EXPERIMENT_CONTRACT.md)

## 要点

本命は「共有する物理ベクトル＋小さなMirrorコード＋非線形処理＋反復」で、複数の有用な機能を少ない実保存量と計算で実現すること。一スカラー方式は別の探索系列として残す。Transformerは検証対象で、最終アーキテクチャの前提にしない。

直接学習、学習済みNNの機械変換、実行系高速化を分離する。独立変数数・数値精度・状態幅・方向数・同時実行数・反復数を混同しない。

最初の作業は既存MA-1162 multiheadのsource／凍結条件／結果を監査して未完部分を閉じること。続いて最小共有核、厳密な共通KV、全体統合の順に進める。新しいMA番号は登録していない。

**主張の境界：** 特定の仮想MLPを一スカラーで実行できても、任意のNNの一スカラー圧縮や独立学習容量の増加は未証明。Mirror並列化で逐次依存は消えず、キャッシュが回転だけで常に再利用できるわけではない。

## 新しい統合仮説

- [Attention・FFN・MoEのMirror共有計算とKV再利用条件](docs/phase2/MIRROR_SHARED_ATTENTION_FFN_MOE_SYNERGY_2026-10-18.md)：同じ共有核から複数機能を取り出し、基底射影・正準KV・注意出力をどこまで一度の計算で共有できるかを段階的に検証する。統合性能は未検証。
