# Mirror-MLP: 現時点で何に使うべきか／どこでは使わないか（2026-10-17）

状態：実験証拠をもとにした**研究用の採用方針**。Transformer正式採用やMLPの汎用圧縮の確証ではない。現時点の比較データは[MLP-ROLE-001](../experiments/mirror_applications/research_intake/mlp_role_20261017/README.md)、[MLP-HYBRID-001](../experiments/mirror_applications/research_intake/mlp_hybrid_20261017/REPORT_JA.md)、[DMFFN-GROK-001](../experiments/mirror_applications/research_intake/dmffn_grokking_20261017/REPORT_JA.md)。

## 結論：Mirrorを使う理由を「複数機能の組み合わせ」に限定し、nativeを第一候補にする

Mirror gate×upは「K_g + K_uの小さいコードからK_g K_u個のアドレスを作る」が、「K_g K_u個の独立MLPと同じ自由度」と同義ではない。MLP-ROLE-001では自然データの数字属性（単一入力から関連する4つの予測）に改善があった。一方、**共有特徴＋4専用出力ヘッド**が保存量・実行時間・品質の強い対照であり、MLP-HYBRID-001でもこれを明確には上回れなかった。

### 主要な証拠（数値は異なる実験設定を横断して比較しない）

|プログラム|成功・失敗の境界|本質的なメッセージ|
|---|---|---|
|DMFFN-001（合成SwiGLU教師）|Mirror整合の9関数で同容量gainに5/5勝ち、未学習gate×up組も改善。無関係関数・自動拡張ではFAIL|組み合わせが再利用できる関数族で有効|
|DMFFN-003（自然画像から独立学習した9MLP）|既知9タスクを少ないバイトで高正解率、**未知3機能は完全共有に負ける**|人工的な合成教師の再利用を自然タスクに外挿できない|
|MLP-ROLE-001（digits属性、表形式回帰、画像復元、時系列）|attrs: Mirror BCE0.113 vs gain0.138、native pair-readout0.103。denoise: Mirror0.0368 vs gain0.0357、readout0.0329。時系列と表形式に差なし|特徴の条件付けには可能性。単純な出力差が主ならreadout優先|
|DMFFN-GROK-001（mod31剰余演算）|Mirror60.0% vs gain35.2%;独立82.3%。遅延汎化5/6、90%到達0/30。WD0では9.2%|Mirrorは規則構造と正則化の相互作用で助ける場合があるが、強いGrokkingではない|
|**MLP-HYBRID-001（自然digits、5新分割＋3長期）**|Mirror+rank2はnative同容量gainに主ゲートFAIL。Mirror+fullはclassificationでnative gain+fullより4/5改善するも、**native pair-readoutより小さくない／安定優位でない**。denoiseではnative優位|private補正を増すだけでMirror優位が成立するわけではない|

## 使い方の条件付き判断規則

1. **最初は最も安価なnativeから**：共有SwiGLU、共有SwiGLU＋pairごとのreadout/bias、FiLM/gain、必要ならrank1/2 LoRAと独立MLPを設置する。baselineを弱くしない。
2. **Mirror候補を限定する**：同一の入力特徴が複数の条件・役割で使われ、**出力の線形読み出しだけでは捕捉できない非線形gate×up相互作用が存在する**ことを開発分割で示す。この段階でもMirror教師だけで評価しない。
3. **入力変換場所を固定**：Mirror座標をSiLU前のgateとupへ分離し、downは共有する。鏡像の変換座標が単なるチャネルgainの再パラメータ化になっていないか、native同パラメータ対照を入れる。
4. **容量を実損失と実バイトで配分**：candidateは `shared readout → per-task bias → task gain/Mirror → rank2 private → full private`。devで同じ学習計算量のloss改善／追加推論bytes、active FLOPs、worst-taskを比較して予算内で選択する。**先にMirror方向数だけを自動増加させない**（DMFFN-001の単純成長FAIL）。
5. **過学習・Grokkingを混同しない**：固定更新数比較とval選択checkpointを両方保存。Grokking研究では遅延汎化の発生時刻を別指標とし、テストピークを選ばない。正則化を両方式で別途校正してからfreshへ進む。
6. **試験通過前はTransformerへ直結しない**：自然MLP群の同バイト・同active compute・同学習FLOPsでnativeに対する品質改善が4/5独立教師世界など再現可能になって初めて試験的にTransformerの最後のFFN層に置く。自然Transformerのhidden stateとKVが影響する場合はキャッシュを勝手に共有しない。

## 次に検証すべき実験

**P0: 条件付き相互作用の欠如がMirrorの負けの原因か**。同じ自然画像の上に、実際の複数属性・機能ラベルに対する共有特徴＋出力ヘッドの残差を分析し、残差がgate×upの低次元座標で減るか試す。nativeのtask-readoutを先に学習した後、「Mirrorを追加することによる差分」だけを実験する。ゲート／bias／rank2残差も同じ追加バイト、同じ学習計算量で付ける。テスト世界は一切モデル選択に用いない。

**P0: 学習済み自然FFNの共有可能性の実際の測定**。別データ・別タスクで学習したSwiGLUのgate/up/downチャネルを*許容される共通置換*で整列する。teacherの出力再現度と、その後の少量データからコードを学習できるかを分離する。oracleの重み合わせで近くても、deploy時の少量データ学習ができなければ採用不可。

**P1: 稀な専門家だけ private、通常役割は共有**。MoEで共通特徴＋pair別readoutから始め、Mirrorで解決できる一部の非線形役割のみMirrorにする。残る役割はprivate fallbackを選ぶ。複数ペアを同じ入力で評価するときだけ大きなgate/up射影を再利用できるので、top-1疎実行で自動速度向上はない。

**P1: MTPドラフト**。共有最後段で位置方向gate×候補プランupを合成し、自然言語の真のp/q検証でaccepted tokens／本体calls／wall時間を測る。4候補の組合せ数を受理数や速度倍率と取り違えない。

## 採用停止条件

- nativeタスク別readoutが同等品質かつより小さい場合、Mirrorを出力分類の標準にしない。
- independent関数でnative/独立MLPから大きく離れる場合、Mirrorコードを増やすだけで万能容量を主張しない。
- 未融合CPUが遅い場合、「理論上の保存容量」を速度向上と呼ばない。
- 3分割・5分割は同一公開データの再分割であり、多様な実世界データセットの再現と混同しない。

## 先行研究（用途の参考でありMirror固有効果の証明ではない）

- Gangwar et al., *Parameter-Efficient Multi-Task Learning via Progressive Task-Specific Adaptation*, AISTATS2026, https://proceedings.mlr.press/v300/gangwar26a.html — 前段共有・後段タスク特化とtask interferenceの整理。
- Qin et al., *Towards Consistent Multi-Task Learning*, CVPR2025, https://openaccess.thecvf.com/content/CVPR2025/html/Qin_Towards_Consistent_Multi-Task_Learning_Unlocking_the_Potential_of_Task-Specific_Parameters_CVPR_2025_paper.html — task特化部分も共通勾配に影響する。
- Hu et al., *Synergistic Intra- and Cross-Layer Regularization Losses for MoE Expert Specialization*, ICML2026, https://proceedings.mlr.press/v306/hu26d.html — 専門家の重複・差別化評価の参考。

**ERROR CHECK：** 有望な用途候補、実際に勝った比較、負けた強対照、計算コスト・過学習・データ制約を分離。研究ブランチのみ更新し、main／旧MAの正式判定を変更しない。