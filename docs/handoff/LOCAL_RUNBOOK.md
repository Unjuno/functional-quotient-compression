# ローカル引き継ぎ手順 — T282

## 0. 今回の状態

この移行は新しい圧縮性能の実験ではなく、コードと証拠の整理である。
計算コア8ファイルは旧ZIPと同一bytes。新しいCPU再生成入口で3候補のファイルSHA256と
復号テンソルhashの一致を確認し、149件の単体テストを実行した。
全27候補の過去の再生成や153テストの記録は保存しているが、今回それら全部を再実行したとは扱わない。

Gitに入るのは現在の小型実行系・主要な結果・出所・引き継ぎ契約。
重み、巨大な圧縮ファイル、重複した全ZIP、旧系列の全runnerはGitに投入しない。
その境界と元ZIPのSHA256は `provenance/t282/SOURCE_ARCHIVES.json` を参照。
元T265の合成gain/compiler系列とT266以降の実モデル系列を混同しない。

## 1. OpenCodeの前に

新規cloneのディレクトリで作業する。公開済みの引き継ぎbranchを指定し、mainと取り違えない。
既存作業ディレクトリでは、まず `git status --short` を確認し、未commit変更を消さない。
実機は「M1 Max・64GBとの申告」であり、chip名、macOS、arm64 Python、利用可能メモリ、MPSを実測する。

```bash
python3 -c "import sys; assert sys.version_info >= (3, 11), 'Python >=3.11 required'"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r experiments/t282/requirements-local.txt
python scripts/fqc_preflight.py --output runs/preflight-001.json
python -m pytest -q experiments/t282/tests
```

依存導入はこのvenv内に限定。既存環境を一括upgradeしない。
このT282ローカル入口はPython 3.11以上が必要（既存core packageの要件とは別）。
測定元はPython 3.13.5 / torch 2.10.0+cpu / NumPy 2.3.5等。
macOS wheelの利用可否や別BLASでのbit一致は未検証である。
pipの失敗を無視せず、導入できた正確な版をfreezeして残す。
OpenCodeのprovider/APIキーの設定はユーザー自身が行い、エージェントへ秘密値を渡さない。

## 2. 入力モデル

利用者提供の `28M(1).zip` の28Mフォルダを `models/28M/` に配置する。
期待するファイルはconfig.json、pytorch_model.bin、tokenizer.json、vocab.json、merges.txt等。
`models/28M/pytorch_model.bin` は次のimmutable lockのhashに一致する必要がある。

`experiments/t282/locks/frozen_candidates.json`

モデル重みはGitにも今回の小型手渡しZIPにも含めない。
Hubからの再取得でhashが異なる場合は入力revisionの違いを調べる。古い期待hashを変更しない。

## 3. 最初に実際に実行するもの

```bash
python scripts/fqc_rebuild_frozen.py --models-root ./models --output runs/rebuild-001
```

3候補を生成し、独立decoderとの一致と過去のartifact/tensor hashを比較する。
出力先が存在すると停止する。これがPASSでも、新しい文章での品質や公式実装との一致は未確認。
圧縮器はNumPyのFP64計算を使うためCPUに固定した。MPSは後で評価経路だけを検証する。

## 4. OpenCodeへの初回依頼

リポジトリrootで `opencode` を起動し、`docs/handoff/OPENCODE_PROMPT_JA.md` を読んで実行するよう依頼する。
AGENTS.mdはプロジェクト規則。opencode.jsonは控えめな権限設定であり、OS隔離の代用ではない。
`--auto` や全権限allowへ切り替えて安全確認を迂回しない。
初回はpreflightとCPU再生成から始める。新しい共有法や64x sweepから開始しない。

## 5. 成果を持ち帰る

`runs/<run_id>/` に、environment.json、protocol.json、inputs.json、commands.log、
results.jsonl、summary.json、failures.json、artifact_manifest.json、NEXT_ACTION.mdを置く。
未完了・失敗・blockedも明記し、raw結果と最終的な解釈を分ける。
最終testを一度評価した後は、そのtestを選択用に戻さない。
エージェントから返す報告は、実行コマンド・完了条件・失敗理由・Git差分・次の一手が追える形にする。

## 6. 外部資料（仕様確認用。研究結果の代替ではない）

- OpenCode rules: https://opencode.ai/docs/rules/
- OpenCode permissions: https://opencode.ai/docs/permissions/
- OpenCode CLI: https://opencode.ai/docs/cli/
- PyTorch MPS: https://docs.pytorch.org/docs/stable/notes/mps.html
- MPS environment variables: https://docs.pytorch.org/docs/stable/mps_environment_variables.html

MPSのCPU fallbackは実行できる演算の範囲を変える。fallbackを使った測定をnative MPS性能として報告しない。
