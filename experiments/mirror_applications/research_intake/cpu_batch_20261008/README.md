# 実際に実行したCPU実験群（2026-10-08）

**保存先:** 独立研究ブランチ `research/mirror-cpu-experiment-batch-20261008`。
**研究上の位置づけ:** MA-1171（圧縮コーデック）とMA-1183（TabM型fast-weight）に対応する、事前登録済みの合成＋実データの機構スクリーニング。**元論文の実装・モデルを再現したものではなく、各MAの科学的statusはUNTESTEDのまま。**

## 結果と実験項目

- [REPORT.md](REPORT.md) — 3つの実験系列の比較・反例・判定。
- [MA1171_FROZEN_PROTOCOL.json](MA1171_FROZEN_PROTOCOL.json) — 圧縮/直列化/全探索の凍結実験条件。
- [MA1183_FROZEN_PROTOCOL.json](MA1183_FROZEN_PROTOCOL.json) — TabM型8-memberの凍結学習条件。
- [MA1183_REAL_TABULAR_FROZEN_PROTOCOL.json](MA1183_REAL_TABULAR_FROZEN_PROTOCOL.json) — 公開2データセットの別凍結条件。
- [AMENDMENT_BEFORE_FRESH.json](AMENDMENT_BEFORE_FRESH.json) — fresh開始前の強い対照群・低レベル復号ベンチの補強。
- [results/MA1171_RESULTS_CORE.csv](results/MA1171_RESULTS_CORE.csv) — 完全な物理保存容量でのPareto比較。
- [results/MA1183_SYNTHETIC_RESULTS_CORE.csv](results/MA1183_SYNTHETIC_RESULTS_CORE.csv) — 5個のfresh世界×3ケース×5モデル。
- [results/MA1183_REAL_RESULTS_CORE.csv](results/MA1183_REAL_RESULTS_CORE.csv) — 実データ2種類×5層化分割×5モデル。
- [results/VERIFICATION.json](results/VERIFICATION.json) — 実行環境・sha256・全シード・再実行・解釈限界。
- [REPRODUCE.md](REPRODUCE.md) — ソース/テスト/再実行。
- [check_batch.py](check_batch.py) — 完全チェックアウトでもネット不要な読み取り専用監査。

## 科学的に解釈できること

1. MA-1171のsource-trained合成Orbit-alignedタスクでは、厳しい実保存容量上限でMirror方式を含む混合配置が非Mirrorの最適配置より低いheldout予測誤差を出す。ただし、off-orbit、容量十分の条件、cold decode速度では一般化しない。
2. MA-1183のrank-one Multi-member合成学習では、Mirrorの小コードは同数の係数を持つ固定線形コードより少し良いが、簡単な1-path共有モデルが総品質・バイト・CPU時間を支配する。構成自体はTabM原論文の公式モデルではない。
3. 追加のscikit-learn癌/ワイン二値タスクでも実データの品質シグナルは測れたが、Mirrorが強い非Mirror対照に対して実用的なPareto改善を示したとは言えない。多数のtest splitは重なり、独立した10世界ではない。

すべての生データ行と失敗結果を残し、workerキュー・`main`・既存公式claim ledgerには何も反映していない。科学的なMA採否の変更には、ネイティブ実装の検証と独立反復が必要。
