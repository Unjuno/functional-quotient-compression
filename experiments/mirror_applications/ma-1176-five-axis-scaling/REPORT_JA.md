# MA-1176 — 5つのスケーリング軸の極小実験・結果

**位置付け:** 4次元・3規則の合成回帰、各世界3 seedの小規模スクリーニング。Attention・自然言語学習・最適アーキテクチャの実証ではない。事前登録の合否条件を変更していない。

## 1. 何を独立にしたか

| 軸 | 独立変数 | 固定条件 | 目的 |
|---|---|---|---|
| Loop | 同一32係数で反復1/2/3回 | 入力分布・係数数・320更新 | 計算の深さと推論蒸留 |
| Mirror | 同幅FiLMと同数の88学習係数 | データ・240更新 | 共有核の機能変換の効用 |
| 物理係数 | 幅4/8/12・学習係数44/88/132 | データ・240更新 | 実係数のスケーリング |
| データ | 異なるユニーク例96/384/1536と汚染/偏り | 同一240更新×32例 | 情報量・質の効果 |
| 探索 | 試行モデル1/4/8 | 合計候補更新数を対照へ投資 | 開発選択対追加学習 |

## 2. Mirrorによる共有関数生成 vs 同予算制御

| 教師 | 方式 | 物理係数 | 記録B | Fresh NMSE↓ | OOD NMSE↓ | 1例の線形MAC下限 |
|---|---|---:|---:|---:|---:|---:|
| Mirror整合教師 | shared | 64 | 272 | 0.02714 | 0.05916 | 64 |
| Mirror整合教師 | film | 88 | 368 | 0.02517 | 0.05258 | 64 |
| Mirror整合教師 | mirror | 88 | 368 | 0.02236 | 0.04277 | 64 |
| Mirror整合教師 | independent | 96 | 400 | 0.02679 | 0.05890 | 32 |
| 独立MLP教師 | shared | 64 | 272 | 0.75868 | 0.78520 | 64 |
| 独立MLP教師 | film | 88 | 368 | 0.07719 | 0.14744 | 64 |
| 独立MLP教師 | mirror | 88 | 368 | 0.25669 | 0.26194 | 64 |
| 独立MLP教師 | independent | 96 | 400 | 0.02508 | 0.05113 | 32 |

Mirror自体のPASS判定: **2/6、FAIL**。Mirror整合教師の改善だけをMirror固有の汎化優位とは呼ばない。

## 3. 物理係数数のスケーリング

| 世界 | 方式 | 幅 | 物理係数 | Fresh NMSE↓ |
|---|---|---:|---:|---:|
| aligned_mirror_teacher | film | 4 | 44 | 0.03477 |
| aligned_mirror_teacher | film | 8 | 88 | 0.02517 |
| aligned_mirror_teacher | film | 12 | 132 | 0.02106 |
| aligned_mirror_teacher | mirror | 4 | 44 | 0.03110 |
| aligned_mirror_teacher | mirror | 8 | 88 | 0.02236 |
| aligned_mirror_teacher | mirror | 12 | 132 | 0.01549 |
| independent_mlp_teacher | film | 4 | 44 | 0.24812 |
| independent_mlp_teacher | film | 8 | 88 | 0.07719 |
| independent_mlp_teacher | film | 12 | 132 | 0.04662 |
| independent_mlp_teacher | mirror | 4 | 44 | 0.50778 |
| independent_mlp_teacher | mirror | 8 | 88 | 0.25669 |
| independent_mlp_teacher | mirror | 12 | 132 | 0.14303 |

3点だけからスケーリング指数や単調な改善法則は主張しない。幅に比例してモデル係数と線形MACも増える。

## 4. データ量と情報品質（更新回数は同一）

| 教師 | 方式 | ユニーク学習例数 | Fresh NMSE↓ |
|---|---|---:|---:|
| aligned_mirror_teacher | film | 96 | 0.02890 |
| aligned_mirror_teacher | film | 384 | 0.02517 |
| aligned_mirror_teacher | film | 1536 | 0.02456 |
| aligned_mirror_teacher | mirror | 96 | 0.02618 |
| aligned_mirror_teacher | mirror | 384 | 0.02236 |
| aligned_mirror_teacher | mirror | 1536 | 0.02371 |
| independent_mlp_teacher | film | 96 | 0.08549 |
| independent_mlp_teacher | film | 384 | 0.07719 |
| independent_mlp_teacher | film | 1536 | 0.06749 |
| independent_mlp_teacher | mirror | 96 | 0.32203 |
| independent_mlp_teacher | mirror | 384 | 0.25669 |
| independent_mlp_teacher | mirror | 1536 | 0.21919 |

品質の操作： `clean_balanced`, `skew`（80/10/10）, `corrupted`（学習ラベル20%シャッフル）, `oracle_filtered`（汚染ラベルを正解メタデータで完全識別し削除）。Oracleは上限対照であり、現実のクリーニング手法ではない。

| 教師 | 方式 | データ条件 | 有効サンプル数中央値 | Fresh NMSE↓ | 希少rule1 NMSE↓ |
|---|---|---|---:|---:|---:|
| aligned_mirror_teacher | film | clean_balanced | 384 | 0.02517 | 0.02898 |
| aligned_mirror_teacher | film | skew | 384 | 0.03003 | 0.04092 |
| aligned_mirror_teacher | film | corrupted | 384 | 0.08354 | 0.07364 |
| aligned_mirror_teacher | film | oracle_filtered | 305 | 0.02444 | 0.02655 |
| aligned_mirror_teacher | mirror | clean_balanced | 384 | 0.02236 | 0.02528 |
| aligned_mirror_teacher | mirror | skew | 384 | 0.02921 | 0.04818 |
| aligned_mirror_teacher | mirror | corrupted | 384 | 0.07360 | 0.07200 |
| aligned_mirror_teacher | mirror | oracle_filtered | 305 | 0.02428 | 0.02423 |
| independent_mlp_teacher | film | clean_balanced | 384 | 0.07719 | 0.09471 |
| independent_mlp_teacher | film | skew | 384 | 0.35094 | 0.54441 |
| independent_mlp_teacher | film | corrupted | 384 | 0.16399 | 0.12337 |
| independent_mlp_teacher | film | oracle_filtered | 309 | 0.06601 | 0.09286 |
| independent_mlp_teacher | mirror | clean_balanced | 384 | 0.25669 | 0.16569 |
| independent_mlp_teacher | mirror | skew | 384 | 0.50569 | 0.53394 |
| independent_mlp_teacher | mirror | corrupted | 384 | 0.36262 | 0.23875 |
| independent_mlp_teacher | mirror | oracle_filtered | 309 | 0.28783 | 0.16359 |

Oracleフィルタの汚染データに対するペア勝ち: Mirror 6/6、FiLM 6/6。5/6の閾値に対してMirror PASS, FiLM PASS。

## 5. Loop：学習の深さと実行時の深さを分ける

| モデル | 物理係数 | 1回推論の線形MAC | Fresh NMSE↓ | OOD NMSE↓ |
|---|---:|---:|---:|---:|
| T1 | 32 | 32 | 0.00919 | 0.02655 |
| T2 | 32 | 48 | 0.00919 | 0.02402 |
| T3 | 32 | 64 | 0.00666 | 0.01604 |
| onepass | 32 | 32 | 0.00520 | 0.01129 |
| distill1 | 32 | 32 | 0.00863 | 0.01986 |

Loopの3段改善（T3がT1を10%以上上回る）：3/3 → PASS。蒸留T1がT3品質5%以内かつMAC半分以下：1/3 → FAIL。蒸留教師の320更新費用は別に必要。

## 6. 探索スケーリング：候補探索 vs 単純な追加学習

| 試行候補数 | DEV最良を選んだFresh NMSE↓ | FiLM追加学習のFresh NMSE↓ | 候補学習秒中央値 | 対照学習秒中央値 | 選択で5%超勝ち |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.06535 | 0.06541 | 0.183 | 0.178 | 3/6 |
| 4 | 0.03809 | 0.03755 | 0.739 | 0.748 | 3/6 |
| 8 | 0.03674 | 0.02912 | 1.485 | 1.429 | 0/6 |

4候補探索が同じ合計更新数のFiLM継続学習に対して5%以上勝つ条件：3/6 → **FAIL**。候補ごとに実MACと保存量が違うので、更新数一致は計算量一致を保証しない。なお探索候補は独立seedも含み、純粋なトポロジー探索ではない。

## 7. 統合した研究上の判断

**事実／仮説の境界:** 今回確定するのは各軸の「物理係数数・実行段数・データ量・試行費用の分離」と、その小型合成課題における品質比較であり、5軸の最適な比率ではない。Mirror数は独立した学習パラメータ数ではない。Loopの層数は動的演算数であり、推論時ゼロ追加Loopに蒸留できるかは別判定。データの情報量は単純な件数と一致しない。

次の段階では、最も改善余地のある軸を2つだけ組み合わせた交互作用実験を行い、強い非Mirror・非Loop制御と総学習FLOPsを揃えて検証する。自然文Attentionへの導入前に、因果性・KVキャッシュ・学習時の総演算量を別ゲートで確認する。

## 8. 主な事後解釈（事前合否を変更しない）

1. **Loopは計算表現力を上げるが、推論Loopの必要性は別。** 3ステップ反復は1ステップ反復に3/3で勝った一方、通常32係数1回の二層MLPがT3の中央値0.00666より良い0.00520だった。これは「反復を別の静的関数へコンパイルする」可能性を示唆するが、指定した教師蒸留経路のPASSは1/3しか達成されなかった。T3優位を全対照への優位へ拡張しない。
2. **Mirrorは独立した関数を無料で増やすわけではない。** Mirrorに整合した教師の中央値0.02236対FiLM0.02517は小規模では好材料。ただし独立MLP教師ではMirror0.25669対FiLM0.07719、独立MLP0.02508。3規則のHouseholder軸は大きさを変えても同じ変換となる（ゲージ冗長性）ため、保存係数88の一部は独立した関数自由度ではない。
3. **データ量とデータの種類は別。** 同じ240更新でユニーク例を増やしても、Mirror整合教師のMirrorは384→1536で中央値0.02236→0.02371とわずかに悪化した。情報量が上がれば必ず良くなるわけではない。また希少ruleの損失は偏ったデータで上がり、ラベル汚染を知るoracleによる除去はどちらの学習方式にも効いた。
4. **探索は構造の適合性を見分けられても、計算効率の優位は別。** 4候補のDEV選択はMirror整合教師で3/3 Mirror、独立MLP教師で3/3独立MLPを選んだ。しかし合計480候補更新とFiLM480継続更新で比較すると探索の品質ゲートは3/6 FAIL。候補数8では0/6になった。探索先の構造によって保存量も演算も違うので、DEV選択の正しさは真の計算効率を意味しない。
5. **次は独立軸を2つだけ交差させる。** データ特徴でMirrorを選ぶ条件付き経路（データ×Mirror）、多段教師から静的Graphへ変換する経路（Loop×探索）を別々に設計する。最大32〜132係数の極小実験が望ましい。Attention＝入力依存MLPへ進むのは両者の因果効果と計算量を確認した後とする。

**ERROR CHECK:** 本研究は一般的なべき乗則、Shannon情報の増加、仮想重み数＝独立自由度、Mirrorの一般的優位、自然言語での学習高速化を証明していない。保持している評価値は3seed×2教師の記述統計。反復数を変えた方式は学習時の実演算が等しくない。プログラム本体・PyTorchランタイムは共通デコーダとしてweight packetへ含めず、モデルごとの学習パラメータ・ルールコードはすべて含める。

## 9. 完全性・再現性

`PROTOCOL.json` は訓練前に凍結。`results/SEALED_TRAIN_MANIFEST.json` の全checkpoints SHA256を評価開始時に検証し、DEV出力を全件照合してからFresh/OODを評価。`RESULTS_CORE.csv` に189モデルの各結果、学習時間・MAC・物理係数・保存量を保存。3seed×2教師なので因果一般化や統計的有意差は主張しない。CPUクロック固定なし。

