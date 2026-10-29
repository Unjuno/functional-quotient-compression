# MA-1181：NoPE × RoPE × Mirror配置の探索的比較

## 目的と判定
主目的は、RoPEがMirrorの追加価値を阻害していたのかを、同じ2回共有AttentionブロックでNoPEとRoPEを比較すること。MA-1178～1180のMLP実験にRoPEはないため説明対象外。

**探索的な主判定**：NoPEでMirror-QのFiLMに対する改善がRoPEより0.02 nat以上増えたseed：**0/3**（事前閾値2/3）。
**NoPE上のMirror-Q有用性**：FiLMを0.02 nat以上改善したseed：**0/3**（事前閾値2/3）。

## Fresh指標（3seed中央値）

| 課題 | 位置方式 | 追加方式 | Fresh CE↓ | Fresh Acc↑ | Batch64 forward μs | Train s | 推論packet bytes | 学習係数 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| set_membership | nope | shared | 0.000361 | 1.0000 | 1137.4 | 1.71 | 15155 | 2450 |
| set_membership | nope | film_q | 0.000361 | 1.0000 | 1294.9 | 1.77 | 15536 | 2482 |
| set_membership | nope | mirror_q | 0.000357 | 1.0000 | 1357.5 | 1.88 | 15536 | 2482 |
| set_membership | nope | mirror_residual | 0.000309 | 1.0000 | 1431.5 | 2.00 | 15536 | 2482 |
| set_membership | rope | shared | 0.000351 | 1.0000 | 1479.5 | 2.09 | 15155 | 2450 |
| set_membership | rope | film_q | 0.000351 | 1.0000 | 1877.9 | 2.42 | 15536 | 2482 |
| set_membership | rope | mirror_q | 0.000350 | 1.0000 | 2017.2 | 2.29 | 15536 | 2482 |
| set_membership | rope | mirror_residual | 0.000344 | 1.0000 | 2530.4 | 2.59 | 15536 | 2482 |
| last_slot | nope | shared | 0.056709 | 0.9844 | 1137.1 | 1.61 | 15155 | 2450 |
| last_slot | nope | film_q | 0.045116 | 0.9922 | 1166.9 | 1.80 | 15536 | 2482 |
| last_slot | nope | mirror_q | 0.025738 | 0.9941 | 1384.7 | 1.95 | 15536 | 2482 |
| last_slot | nope | mirror_residual | 0.021558 | 0.9961 | 1386.9 | 1.99 | 15536 | 2482 |
| last_slot | rope | shared | 0.000365 | 1.0000 | 1699.7 | 2.27 | 15155 | 2450 |
| last_slot | rope | film_q | 0.000365 | 1.0000 | 1854.9 | 2.37 | 15536 | 2482 |
| last_slot | rope | mirror_q | 0.000302 | 1.0000 | 1962.5 | 2.58 | 15536 | 2482 |
| last_slot | rope | mirror_residual | 0.000521 | 1.0000 | 1982.2 | 2.43 | 15536 | 2482 |

## 対応seedの交互作用（Fresh CE）

| 課題 | seed | NoPE gap (FiLM−MirrorQ) | RoPE gap | 差分（NoPE−RoPE） | 主ゲート |
|---|---:|---:|---:|---:|---|
| set_membership | 101 | 0.000002 | 0.000010 | -0.000008 | FAIL |
| set_membership | 211 | 0.000015 | 0.000004 | 0.000011 | FAIL |
| set_membership | 307 | 0.000003 | -0.000000 | 0.000003 | FAIL |
| last_slot | 101 | -0.007904 | 0.000062 | -0.007967 | FAIL |
| last_slot | 211 | 0.034374 | 0.000016 | 0.034358 | PASS |
| last_slot | 307 | 0.000015 | 0.000001 | 0.000015 | FAIL |

## 制約・研究解釈

- NoPEは明示的な位置符号化を与えないが、因果マスクと2回の共有Attentionから位置依存性が間接的に学習され得る。位置依存課題と位置非依存課題を分けて報告した。
- QだけのMirrorとQ/K両方の同一Mirrorは異なる。後者はNoPEではQK内積を変えず、RoPEで非可換なら変える。2D厳密検算は algebra_probe.py。
- 一つの固定ViewだけのQ変換はQの投影重みに吸収可能であり、今回の追加自由度は2回のLoopで別コードを持つ点にある。
- 位置非依存のset_membershipでは大半の方式が高精度に到達し、性能差を検出しにくい。NoPE有利・不利を一般化しない。
- Mirror-Q/FiLMは同数の追加コード係数、Mirror-Residualも同じコード数だが活性化される位置・初期摂動が異なり完全な関数差統制ではない。
- CPU FP32 1 threadでのPython eager実装。学習更新数のみ一致、真のFLOPs・GPUカーネル最適化は一致しない。Mirror自体は追加反射演算を使う。
- 新規3seed・2合成課題のスクリーニングであり、確認実験、自然言語NLL、一般LLM学習効率、KV cacheを実証しない。
- 追加のRoPE周波数最適化、混合NoPE、ALiBiは比較していない。一般的NoPE改善を主張しない。

## 監査

- 48モデルのチェックポイントSHA256検証: 48 件。
- Fresh出力再生 48 件、最大絶対CE差 0。
- 学習前にGitHub別研究ブランチへ README.md、PROTOCOL.json、STATUS.md、run.py、test_run.pyを登録した。旧実験の判定は変更しない。
- プロトコル SHA256: 5160b6a7fa756b14a434cb50e8afa00da2a48f82da09c34d5fa9c071a1fb30dd。
- 学習済み manifest SHA256: 2f6ddf881007a043c7188a73b230f98f6bbaa97e8cfeee1f22ee9cf22a192e3a。
- 実行環境: {'cpu_model': 'AMD EPYC 9V74 80-Core Processor', 'cpu_mhz': '2596.146', 'torch': '2.10.0+cpu', 'python': '3.13.5', 'platform': 'Linux-6.18.55-x86_64-with-glibc2.41', 'threads': 1}

## 次の反証課題

位置非依存課題をもう少し難しくし（長距離照合など）、RoPE・NoPE・RoPE/NoPE混合Attentionと、Q-Mirror／Residual-Mirror／正規化した学習可能Q変換を同予算で比較する。位置依存タスクを失う代償が大きいならNoPE全面採用はしない。
