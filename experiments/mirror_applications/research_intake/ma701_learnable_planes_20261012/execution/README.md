# MA-701-LP01 Phase A — 学習可能な共有Mirror軸の実行結果

2026-10-12。**主計画: FAIL_DEV_GATE。追加の事前固定したLR診断: 同族の合成関数で限定的な品質/保存量の改善。実画像での優位性は未成立。** 元のfresh17111–17115は未開封。Phase B/C、公式MA-701本体の完了、自然言語モデルの成功を主張しない。

## 実装と証拠

Mirror insertion: one shared FFN receives a four-angle task coordinate and four shared learned plane-vector pairs before GELU, instead of allocating a full task FFN. The shared FFN, plane vectors and task codes train jointly from task labels. No alignment loss or private residual was added to the primary candidate.

今回は**実際に動かしたソースコードとテストを、このexecution/source・execution/testsへ公開**した。従来の要約だけの記録ではない。完全な356チェックポイント・生ログは会話の証拠ZIPに保持し、ソースから再生成可能。学習用コードの事前ハッシュはPRE_RUN_LOCK.json、具体的条件はPROTOCOL.json、再実行はREPRODUCE.md。

実行: 128校正ジョブ+192開発ジョブ=320学習ジョブ、313,600 optimizer updates。長期チェックポイントを含め356保存モデル、別途2回の完全再学習3,200 updates。45単体テスト、7,150品質スカラーの再計算は誤差0。2完全再学習のNPZ SHA256も一致した。

## 1. 主計画（開発seed17101–17103）

8設定の校正で学習軸LR/開始step/初期角度を選び、nativeにも最大8設定を提供。学習軸の共有重みLRは元計画通り0.001固定、nativeは0.01まで選択できた。`fixed_matched`は学習軸と全初期値・LRを揃え、軸だけ凍結する直接対照。`fixed`は別途校正した強い固定軸対照。

800更新、3seed平均:

| task | fixed_matched | learned planes | native reference |
|---|---:|---:|---:|
| aligned synthetic nMSE | 0.277017 | 0.241609 | LoRA2 0.105121 |
| partially shared synthetic nMSE | 0.385987 | 0.353836 | LoRA2 0.201140 |
| unrelated synthetic nMSE | 0.835996 | 0.798100 | private FFN 0.251854 |
| digits8 BCE nat/label | 0.122667 | 0.104872 | ordinary multihead 0.075251 |

相対的な軸学習効果はあるが、合成絶対nMSE<=0.05と画像のseed別品質条件を満たさずFAIL。freshを開かず、旧判定を修正しない。

## 2. 追加LR診断（別の新規開発seed27101–27103）

主結果を観測後、別プロトコルをコミット2687f540cee12f97c60449c690899fd84310ab1fで固定。共通LR=軸LRを0.001/0.003/0.01、初期角度標準偏差0.03/0.12 radの同じ6設定から校正17100のvalidationだけで選択。nativeの既存一致設定を再利用し、未学習の候補8件を追加。選択を4a44259b2c2b9bed3cfa0874c0a92329f4e49c0fで固定してから新規seedを実行した。これは追加開発診断であり、正式fresh追試ではない。

### 合成課題：回転構造が実在するときは改善

12個の非線形関数、入力24、隠れ32、出力6、各タスクtrain128/validation128/test512例。教師の共有回転面・重み・コードは渡していないが、**教師は候補の表現族に整列して作っている**。

| method | nMSE mean at800 | nMSE mean [min,max] at2400 | complete FP32 NPZ |
|---|---:|---:|---:|
| fixed, same hyperparameters | 0.178375 | 0.145197 [0.124960,0.171143] | 11,448 B |
| learned shared planes | 0.080823 | **0.018660 [0.016378,0.019985]** | **11,450 B** |
| LoRA rank2 | 0.110005 | 0.074224 [0.062533,0.083889] | 15,380 B |

同じ2,400更新のLoRA2よりnMSE約74.86%減、保存量25.55%減を3seedすべてで確認。合成で選ばれた設定は共有/軸LR0.01、初期角度標準偏差0.12、warmup0。改善全部をLRだけに帰属させない。固定面は可換な座標面、学習面は非可換になれるため、その自由度の差も含む。

### 実画像：改善は安定せず、長期学習で過学習

1,797枚の手書き数字画像、60/20/20画像ID分割。同じ入力の偶奇・大小・素数等8二値ラベル。8つの独立自然タスクではない。画像校正では元のLR0.001、角度標準偏差0.03が選ばれた。

| method | BCE at800, 3seed mean [min,max] | BCE at2400 | accuracy at800 | complete bytes |
|---|---:|---:|---:|---:|
| fixed_matched | 0.109062 [0.094045,0.117413] | 0.157219 | 96.39% | 31,294 |
| learned planes | 0.126361 [0.087865,0.159951] | 0.267873 | 96.63% | 31,296 |
| native multihead | **0.077962 [0.075145,0.080873]** | not run | 97.28% | **30,344** |
| FiLM | 0.089211 [0.056771,0.109139] | 0.105805 | 98.44% | 31,725 |

学習軸のtrain BCEは0.030994→0.000193へ改善した一方、validationは0.158430→0.364503、testは0.126361→0.267873へ悪化。したがって画像で単純に更新数やLRを増やす根拠はない。次はvalidationによる停止/チェックポイント選択をnativeと同条件で検証し、正則化・軸数増加・専用残差の効果を混ぜずに調べる。これらの改善はまだ未実行。

## 3. 計算コストと限界

CPU AMD EPYC9V74、clock snapshot2.596GHz（未固定）、quota4 cores、4GiB、Torch2.10.0+cpu FP32、one thread。合成batch64学習、画像batch128学習。推論計時は学習終了後に単独実行、15warmups、7repetitions×10calls、batch1/64、全View出力。範囲は3seedの観測値で信頼区間ではない。

合成追加診断の推論中央値: eager B1学習軸0.2630ms vs LoRA2 0.0684ms、B64 0.4352ms vs0.2719ms。eager速度優位はない。事前展開後はB1 0.0422ms vs0.0409ms、B64 0.1032ms vs0.0986ms。ただし学習軸の保持テンソルは9,440B→47,904Bへ増える。保存ファイルと展開キャッシュを同じメモリとして数えない。通常多ヘッドは共通FFNを1回だけ計算する専用展開経路で比較した。

固定面はこの共通NPZ形式で明示保存したため、省略できる定数を除くとさらに安くできる。学習軸が固定軸より小さいとの主張はしない。直交変換はnative二Householder反射でも同じ出力を得られ、回転演算自体は新規ではない。HRA対照は簡略反射対であり論文全体の再実装ではない。全手法の学習・調整時間を測定したが、matched-updatesはmatched-FLOPsでも完全収束でもない。

## 4. 次の調整方針

1. 実画像はvalidationによる早期停止を最初に検証する。最良test点を後付け採用しない。
2. 共有重みLRと軸LRは対象タスクごとに校正する。0.01を普遍的推奨値としない。
3. 固定ランダム非可換軸と学習非可換軸を比較し、可換性と軸学習の寄与を分離する。
4. その後にタスク数・疎選択・privateを別々に検証する。今回Phase B/Cは未実行。

線形代数では共有変換の可換性と自由度、最適化では軸への勾配と学習率、統計学では過学習と再分割依存、システムでは保存量と展開キャッシュの差が重要。

## H/T/D/C/U

H: shared learned axes yield useful low-description task variation.
T: Phase A primary120cal+150dev; diagnostic8cal+42dev, long checkpoints prospective. Tests and full weight replay.
D: primary FAIL_DEV_GATE; diagnostic aligned mechanism promising only; real-data superiority and Mirror-specific novelty unestablished. Original fresh sealed.
C: teacher-family alignment, noncommuting geometry versus disjoint fixed planes, optimization budget and overfit, native equivalent transforms.
U: only3seeds/study, shared digit dataset, unlocked clock, finite samples, explicit FP32 folding error up to6.1e-5 logit; no population CI or general LLM claim.

No main/official registry changes. Standalone container workspace was used because git DNS failed; historical full-repository tests were not rerun. Complete code is in source/, tests/; complete binary models are in the conversation evidence ZIP, not this GitHub directory.
