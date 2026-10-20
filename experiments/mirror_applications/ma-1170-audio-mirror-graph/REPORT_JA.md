# MA-1170 実証報告 — 音声特徴のMirror/Loop仮想グラフ・オートエンコーダー

2026-10-20 JST。**Verdict: FAIL (frozen Mirror-specific quality + localization); pilot scope only.**

## 対象の分離
1. **検証したもの**：16kHz PCMから計算する64帯域・対数Melパワー特徴を、TopK=6の16次元非負コードに圧縮し、4ノード×16本仮想4x4エッジ、2回Loopで復号する方式。
2. **検証していないもの**：生PCMの可逆圧縮、位相の復元、知覚的な音質（PESQ/STOI/MUSHRA）、意味特徴（音素・話者）、異なる音声コーパスへの一般化、大規模ワールドモデル。
3. 同じエッジ・ページ・アドレス・ゲート数の **affine** 対照が主比較で、通常 **sparse AE** と **native hidden AE** が追加対照。

## H — 反証可能仮説
- H1（品質）：全3ペアseedでMirrorのFresh NMSEがaffineより少なくとも5%低く、Mirrorの3-seed中央値がsparseより低い。
- H2（局在）：全3seedでMirrorの潜在座標除去時top8/64帯域集中度がaffineとsparseの双方より絶対値+0.05以上高く、再構成NMSEの劣化が5%以内。
- H3（探索的）：学習済みゲートによる下位8本除去が上位8本除去より損失を増やしにくい。未事前登録の追試なのでH1/H2判定には使用しない。

## T — 最小実験
- 出典音声：pyannote.audio同梱のsample.wav。SHA256   `c319b4abca767b124e41432d364fd7df006cb26bb79d09326c487d606a134e6e`。16kHz/mono/30s; PCM16。
- STFT: Hann400サンプル、hop160、64 Mel、80–7600Hz、log1p(10×Melパワー)。各帯域の平均・標準偏差は訓練時間帯のみ算出。
- 音声時間軸：train [0,17) 秒1698フレーム、dev [18,21) 298、fresh [22,25) 298、late [26,29) 298。境界には1秒の空白。
- 12学習：4モデル×seed=31,37,41。Torch FP32、CPU1スレッド、AdamW lr=.003、weight_decay=.0001、batch=96、各500更新、50更新ごとdev選択。fresh/lateは12モデル選択後のみ評価。
- Mirrorグラフの**独立学習変数は64個**：共通core16・ゲート16・アドレス32。全16本4x4エッジ(合計256仮想重み)は共有coreから毎forward生成。活性はReLU、Loop2。affineと共通のencoder+decoder等を加えると合計2144学習スカラー。
- Mirrorは固定直交Householder→tanh→逆Householderの非線形重みView。affineは同一4ページ、同一アドレス・ゲートに対する直接アフィン合成。
- 全方式の符号長は**16×8=128 bit/フレーム**、約100フレーム/sなら12.8kbps（ただし波形ビットレートとの音質同等比較不可）。
- 保存量はtorch.save推論packet（重み・正規化・量子化スケール・プログラム識別子）と、実書出ししたuint8コード列を合計。実行ソースコードそのものはpacketに含めないため、独立実装配布では追加バイトが必要。

## D — 主要評価（3seed中央値）

| Mode | 学習変数 | Fresh NMSE↓ | 8bit Fresh NMSE↓ | Late NMSE↓ | 潜在top8局在度↑ | Packet+コード（298frames）B↓ | 推論B1中央値µs↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| mirror | 2144 | 0.342928 | 0.344146 | 0.439012 | 0.390951 | 17,649 | 106.71 |
| affine | 2144 | 0.333077 | 0.335699 | 0.437238 | 0.390693 | 17,649 | 90.54 |
| sparse | 2080 | 0.347384 | 0.347604 | 0.439409 | 0.378402 | 16,829 | 17.50 |
| native | 2352 | 0.340326 | 0.340236 | 0.435175 | 0.375524 | 18,293 | 26.59 |

- 比較用の無圧縮FP32 Mel特徴298フレーム=76,288バイト。Mirrorコード+packet=17,649バイト、**約4.32×のMel特徴ストレージ削減（非可逆）**。sparse=16,829バイト、つまりこちらが少量。
- Mirror/affine Fresh NMSE比 [31:1.0296, 37:1.0254, 41:0.9633]。5%以上改善は0/3。MirrorのFresh中央値 0.342928、affineは0.333077。
- H1=**FAIL**、H2=**FAIL**。Mirrorのtop8局在中央値0.390951、affineの0.390693に対して絶対差+0.000258で+0.05に届かない。局在度自体は音素の解釈可能性や概念対応を意味しない。
- 8bit量子化による誤差上昇は主なボトルネックではない。16次元TopKの情報損失・最適化不足・符号化アーキテクチャを切り分ける必要がある。

## 追加した強い単純対照（**事前登録外、探索的**）
- trainだけで求めた**rank16 PCA**はFresh NMSE 0.229338、8bit符号で0.242770。PCA推論packet 7,573B、298フレームのuint8ストリーム4,780B、総計12,353B。さらにB=1推論中央値6.74µs。**Mirrorより高精度・省メモリ・高速**。
- PCAの入力特徴と圧縮次元は同一であり、監査データではなくtrain SVDで学習している。ただし後から追加した対照なので事前登録判定の書換えには使用しない。

## H3追加診断（事前登録外）
- 16本のうちゲート下位8本を推論時遮断：MirrorのFresh NMSE中央値が全辺時0.342928、遮断後0.342791。seed41では損失が0.3628→0.3735と悪化。
- ゲート上位8本・乱択8本も比較（`results/EXPLORATORY_EDGE_MASK.csv`）。ゲート順位は完全には因果重要度を表さない。Mirrorの下位8本遮断が上位8本遮断より良好なseedは2/3にとどまる。

## 独立した可逆比較
- 同一16kHz/mono/30s音声のraw PCM16: **960,000B**、FLAC: **315,107B**、デコード時PCM16サンプル **bit-exact=True**。比率 **3.047×**。
- MirrorのMel特徴圧縮をこのFLAC音質・完全可逆性と同等とみなしてはいけない。STFTからMelバンドへの変換と位相の廃棄だけでも可逆性が失われる。

## C — 破れ方・対立仮説
1. Mirror Viewの非線形性がデータ集合上でほぼ線形にとどまり、同一アドレスaffineに対する追加自由度が小さい。
2. ゲート/アドレスの連立学習が不安定、固定500更新では収束不足。より長期学習と強い単純制御が必要。
3. 周波数帯の局在度は意味解釈可能性ではない。semantic/causal probeが不可欠。
4. 入力フレームごとのAEでは時間相関を明示的に捉えない。時系列ワールドモデル用途には因果的予測・長期状態保持を評価する必要がある。

## U — 不確かさ・測定前提
- 3 seedのMirror−affine Fresh NMSE差は +0.009851, +0.008264, -0.013807。平均差+0.001436。seedばらつき由来の記述的標準不確かさ u_c≈0.007635、k=2の形式的幅≈±0.015270。これは独立コーパス一般化を含まず、95%信頼区間を意味しない。
- 1録音内の連続フレームは強く自己相関。ハード AMD EPYC 9V74、CPU-only PyTorch2.10.0、1スレッド、batch=1（速度）、クロック固定なし；RUNTIME比較は当該コンテナ限定。
- checkpoint→packet再ロード後の出力は全12モデルでbit-exact。MLPの実論理エッジ数がそのまま独立表現能力を意味しない。
- 保存量の誤差要因：torch保存コンテナのメタデータ、実装コードの未計上、量子化の帯域外値、フレームヘッダ。メトリクスに関するu_cの主要要因はseedと録音依存。

## ERROR CHECK
- 構造テスト: pytest **9/9 PASS**。モデル学習 **12/12終了**、学習パラメータ・tensor shape、Mirrorアドレスgradient、非線形差分、exact packet-replayすべてPASS。
- protocol JSON SHA256 `86da7956cfac5eab05eb93be4b15fabeb3c9f46cab4807eb7edf1f3ae431557f`；source WAV SHA256 `c319b4abca767b124e41432d364fd7df006cb26bb79d09326c487d606a134e6e`。全12チェックポイントにprotocol/source/program SHA記録。
- Source/test/weights full reproduction archive: `MA1170_audio_mirror_graph.zip`（会話添付）。GitHubの旧`FROZEN_PROTOCOL_SUMMARY.md`中で言及されたJSON SHAは元JSONが保存されていなかったため独立照合不能。今回のJSONは上記SHAで学習開始前にローカル凍結し、GitHub側の事前登録済み概要と条件を整合させた。
- この報告は2026-10-20の**実測再実行**に基づく。過去の会話で先に示されたFresh NMSE表は根拠ファイルが見つからなかったため、再実行結果と同一視していない。

## References
- Zeghidour et al, SoundStream: An End-to-End Neural Audio Codec, IEEE/ACM TASLP 30, 495–507 (2022), DOI `10.1109/TASLP.2021.3129994`.
- Défossez et al, High Fidelity Neural Audio Compression, arXiv:`2210.13438`.
