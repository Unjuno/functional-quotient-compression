# Vector Mirror：非線形活性化・概念境界・Loop・MoEに関する先行研究調査

**作成日:** 2026-10-28 JST  
**位置付け:** 研究文献・数理診断・次の事前登録案。新しい訓練実験の実施、既存MA判定の変更、新規性の立証を意味しない。

## 1. 研究質問

「一つの共有計算核を、異なるMirror Viewを通して非線形活性化させることで、多数の概念境界と機能を分化させ、さらにLoopで逐次的に利用できるか。物理MoEを増設しない場合でもFiLM・共有低ランク・独立Expertより実用上優位な領域があるか。」

## 2. 原著論文と研究上の位置付け

| 先行研究 | 原典 | 確認できる主張 | Mirrorとの相違・必要対照 |
|---|---|---|---|
| **Conic Activation Functions** | Fu & Cohen, UniReps/PMLR 2024, https://proceedings.mlr.press/v285/fu24a.html | ReLUを非負直交錐への射影と捉え、Lorentz錐などへの投影型多変量活性化CoLUを提案 | 境界幾何の先例。View条件付き・共有核多機能学習とは異なる。CoLUを非Mirror対照に |
| **From ReLU to GeMU** | Li et al., *Neural Networks* 2025, DOI 10.1016/j.neunet.2025.107654, https://pubmed.ncbi.nlm.nih.gov/40516382/ | 二次錐などの多変量射影型活性化の表現利得を理論・実験で議論 | 「錐を使うだけで表現が増える」は新規性ではない。GeMU比較 |
| **Rotate the ReLU to Sparsify Deep Networks Implicitly** | Nayak & Kalyani, TMLR 2024, https://openreview.net/forum?id=Nzy0XmCPuZ, 初稿 arXiv:2206.00488 | 活性化の符号・ゲインを学習し、剪定候補を発見して構造疎化 | 名前が似るが、最終形は成分別変調。高次元の Q^T ReLU(Qh) とは異なる。可変ゲートとの比較 |
| **Householder Pseudo-Rotation** | Pham & Nguyen, EMNLP 2024, DOI 10.18653/v1/2024.emnlp-main.761, https://aclanthology.org/2024.emnlp-main.761/ | LLMの状態の方向をHouseholder型変換で編集し、ノルム整合を意識する | 学習済みモデルの振舞い制御が中心で、反射前後に同一非線形共有核を置くものではない |
| **Householder Reflection Adaptation（HRA）** | Yuan et al., NeurIPS 2024, arXiv:2405.17484, https://proceedings.neurips.cc/paper_files/paper/2024/hash/cdd0640218a27e9e2c0e52e324e25db0-Abstract-Conference.html | 反射の積による直交適応は結合低ランク適応と関係し、モデル適応で効果 | 正式な強対照。単一反射なら両側重みの結合ランク1変化へ代数的に吸収可能 |
| **A lightweight and gradient-stable neural layer（Han-layer）** | Yu & Zhang, *Neural Networks* 2024, DOI 10.1016/j.neunet.2024.106269, https://pubmed.ncbi.nlm.nih.gov/38604008/ | Householder重みとABS活性化を組合せ、単体Jacobianの直交性を確保 | Loopでの勾配安定性の代案。ABSはReLUのような信号切断ではない |
| **Concept Whitening** | Chen, Bei & Rudin, *Nature Machine Intelligence* 2020, arXiv:2002.01650 | 白色化と概念軸への回転により、概念軸の解釈を助ける | 既知概念ラベルによる明示的軸整列であり、自由な機能多重化の性能証明ではない |
| **Polytope Lens** | Black et al., arXiv:2211.12312 | piecewise-linear活性化が分割する多面体領域を解釈単位として分析 | 境界を動かす機構の直接的解釈フレーム |
| **Neural Networks Are Already Clean** | Cammarata, 2026-10, https://already-clean.goodfire.com/ | ニューロンは想定より単一の主要な機能に近いという作業仮説 | 独立確証ではない。弱信号が安全に削れるとの一般結論は禁止 |
| **Neural networks are not already clean** | Leo Gao, 2026-10-17, https://www.lesswrong.com/posts/Dkh7pXKjoNWkkxQDz/neural-networks-are-not-already-clean | 弱い活性化の強度・組合せや下流KLが重要かもしれないと指摘 | 弱信号は振幅閾値のみで削除せず、下流介入で重要性を推定 |
| **Toy Models of Superposition** | Elhage et al., 2022, https://transformer-circuits.pub/2022/toy_model/ | 疎性・非線形性・特徴相関がsuperpositionと干渉へ影響 | 単純な「等角配置が常に最適」を排除。共起と類似性を独立操作 |
| **Shared Weights, Selected Computations** | Wu, Hu & Zhang, arXiv:2609.39892（2026-09-30） | 凍結Loop Transformerの入力状態を線形Jで変えて計算対象を制御。Attentionルーティングへの因果介入 | Loop境界Mirrorの第一候補。Jは対角＋rank48＋biasなのでMirrorより表現力が広い |
| **Scaling Laws for Looped Mixture of Experts** | Chen, Goyal & Krishnamoorthi, arXiv:2609.40316（2026-09-30） | 反復の有効容量増加の飽和とMoE疎性との交互作用を実証的にモデル化 | Mirrorが物理MoEの代替になるとは示さない。Loop費用と経路多様性が必要 |
| **BatchEnsemble** | Wen, Tran & Ba, ICLR 2020, https://research.google/pubs/batchensemble-an-alternative-approach-to-efficient-ensemble-and-lifelong-learning/ | rank1変調による物理重み共有と機能別の利用 | 安価な共有モデル対照 |
| **Superposition of Many Models into One** | Cheung et al., NeurIPS 2019, arXiv:1902.05522 | コンテキストコードで複数モデルを同一重み集合へ格納 | 共有パラメータから多数機能という大枠の先例 |

**新規性に関する結論:** 「ReLUの境界を変更する」「反射を学習する」「共有重みを条件付けする」はそれぞれ先例がある。未検証なのは、*少数の条件付きMirrorコードで共有核の非線形境界を切り替え、同時に学習可能性・機能選択性・実バイト/実行時間のPareto改善を得る領域*である。

## 3. 数理的に重要な設計制約

変数表（量はすべてSI単位1）：

| 記号 | 日本語定義 | 単位 | 定義域・型 |
|---|---|---|---|
| h | 共有核の前活性状態 | 1 | 実d次元ベクトル |
| d | 隠れ次元 | 1 | 正整数・スカラー |
| Q_m | m番目の直交Mirror | 1 | d×d直交行列 |
| m | 機能/View番号 | 1 | 離散整数・スカラー |
| σ | 活性化（ReLU/SiLU/tanh） | 1 | R^d→R^d の成分別写像 |
| Φ_m | Viewごとの実効活性化 | 1 | R^d→R^d の写像 |
| ε | 入力振幅の尺度 | 1 | 正のスカラー |
| W_u,W_d | 共有MLP入出力行列 | 1 | R^{d×p},R^{q×d}の行列 |
| P | 非負直交錐 | 1 | R_{≥0}^d、凸集合 |
| K | Viewの個数 | 1 | 1以上の整数 |

実効活性化を

Φ_m(h)=Q_m^T σ(Q_m h)

と定義する。ReLUでは Φ_m(h) は回転した錐 Q_m^T P へのEuclidean射影であり、活性化境界の方向が変化する。直交変換自体は情報を分離せず、**ReLUがその座標で遮断する**。単一Viewでは W_d Q_m^T と Q_m W_u に吸収できるので一般的Denseより機能クラスが広がるとは限らない。複数条件でコードを共有・制限することが価値の候補である。

εが小さいとき異なるQの出力差は、一般に次の先頭次数になる：
- ReLU：Φ_Q(εh)-Φ_I(εh) = ε[Φ_Q(h)-Φ_I(h)]、厳密な正の一次同次性。
- SiLU(z)=z/(1+exp(-z))：Φ_Q(εh)=εh/2 + ε² Q^T((Qh)⊙(Qh))/4 + O(ε⁴)。View差は通常2次。
- tanh：Φ_Q(εh)=εh - ε³ Q^T((Qh)^{⊙3})/3 + O(ε⁵)。View差は通常3次。

※Qは直交、⊙は成分別乗算。次元は両辺とも無次元のd次元ベクトル。高次項が偶然相殺される入力・Qもある。「一般に」は典型的な非退化配置を指す。

**実測前の小さなFP64診断:** d=2、回転角0.4 rad、h=(0.9,-0.5)。入力振幅を0.1→0.05→0.025→0.0125とすると、View出力差の測定次数はReLU 1.000、SiLU約2.000、tanh約3.000。これは数学機構の数値検算であり、学習・性能・推論速度を測ったものではない。

ReLUだけの場合、Φ_Q は射影なので Φ_Q(Φ_Q(h))=Φ_Q(h)。**同一Qを再度通すだけでは追加処理がない**。異なるQやAttention/Residualの介在があると話が変わる。

## 4. 実験の優先順位（提案・未訓練）

### P0. 非線形Mirrorの機構を封印前に検査
- 各活性化でQへの勾配、View間出力差、錐の有効境界数、固定Viewの同値性を監査。
- ReLU、SiLU、tanh、任意でABS/CoLU/GeMU。
- 校正入力は学習実分布から別seedで採取した無ラベル入力。Mirrorと全対照の初期関数変化量を同じ校正規則で評価。校正不能なら事前停止。
- 検証：FP64有限差分、異なるviewでの非恒等性、軌道上の活性化領域占有率、実測バイト・計算費用。

### P1. 共有MLPの機能分化：**MoEなし**
- 2～4Viewで、類似性×共起率を独立に変えた合成教師を主課題とする。通常Dense・FiLM・BatchEnsemble・結合low-rank・HRA・固定Q・ランダムQ・独立MLPを対照にする。
- 既知コードと未知コード、凍結核と共同学習核を交差させ、表現限界/最適化限界を分ける。
- 弱信号については、振幅閾値に加え、抑制前後の損失・下流KL、介入での機能選択性を測定。**弱い信号を消すだけでなく、別Viewに移管した価値を測る**。
- 目標は、同じ実バイト・総学習FLOPsの最強非Mirror対照に対して独立Fresh/OODで改善すること。

### P2. Shared Transformer + Mirror-Loop：**まだMoEなし**
- 独立に学習した小型共有Transformerブロックへ、Loop境界でQ_t / Q_t^{-1} を挿入。Wuらの J（diag+low-rank+bias）を必須対照にする。
- グラフウォークの異なる次ステップを、View交換・アブレーションで因果的に選べるか。Attention pattern patchingも利用。
- R=1/2/4/8, K=1/2/4、必要時1Viewのみ実行する対照と全View実行を区別。共有上射影以外の計算は無料扱いしない。

### P3. MoE併用（P1/P2で固有優位が示された場合のみ）
- 物理Expertを増設する前に、共有核＋Mirrorだけで既存Expertの関数多様性を何割代替できるかを判定。
- 実体Expert追加、共有＋FiLM、共有＋低rank、共有＋Mirrorを同保存量・同アクティブFLOPsで比較。

## 5. 反証可能な研究判断
- H: 実入力上での有効活性化境界の増加が、同資源の非Mirror対照では得られない一般化・機能分化に対応する。
- T: 開発5生成seed、独立確認最低10生成seed、評価後の設定変更を禁ずる。Freshはチェックポイント・コード・プロトコルSHA封印後のみ生成。
- D: 例としてFreshの対応誤差比≤0.90を事前効果量とし、信頼区間やseed勝率も併記。OOD劣化、bytes/runtimeの悪化は別ゲート。**効果量・サンプル数は訓練前に本登録**。
- C: (i) 低rank両側変更に厳密等価、(ii) ReLU/SiLUの非線形性による一般利益、(iii) 共有による統計利得、(iv) 追加計算、(v) 事後的初期化校正がMirror差を説明する可能性。
- U: 主な不確かさ＝生成関数族、活性化近線形性、ルータの情報漏洩、seed、GPU kernel、実計算量proxy。対応測定の標準誤差u_cとk=2を報告するが、少数seedの正規近似を95%保証とは呼ばない。

## 6. 結論

「Mirrorで活性化境界を変えられる」は厳密な数学的機構だが、それ単体は新規ではない。最も価値がある未検証部分は、**少数の動的Viewを一つの物理共有核と共同学習し、必要なときだけViewを切り替えて、その非線形フィルターを複数の意味機能へ活用できるか**。まずReLU/SiLUの小型モデルで既存のtanh問題を切り分け、続けてLoop境界で検証し、MoEは最後に追加する。

注意：今回の調査では大規模訓練を実施していない。外部データに基づく優位性は原著者の報告であり、独立再現ではない。
