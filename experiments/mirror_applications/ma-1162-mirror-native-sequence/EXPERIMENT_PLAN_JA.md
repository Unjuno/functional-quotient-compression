# MA-1162 実験計画 — Mirror中心の因果系列モデル

**主眼を1つに絞る：** 「共有した物理トークンミキサーにK個の非線形Mirror方向を追加すると、**本当に異なる因果系列の処理**を学習でき、同バイト・同演算の普通のモデルより有益か」。

## 0. 既存研究と設計上の制約

- MA-041/249：Mirror head の適合合成教師での保存効果はあるが、自然な構造・GPU時間は未知。
- MA-247/431：共有ブロックの反復は学習が難しく、明示的な通常重み共有の対照が必要。
- TM001/MTPM-001：並列未来出力は実現可能でも、未観測の潜在分岐では系列整合性が壊れる。
- MA-691/253：KVは条件付きでだけ共用可能。中間FFN Viewの変更は後段キャッシュを変える。
- MA-1156/1157：非線形Mirrorと同コード反復の研究は既にあるので、新規寄与は**因果トークンミキシングと平行View×反復の共同構造**。
- MA-597～604：巨大仮想重み/ハッシュ/動的重みに失敗例あり。超高精度スカラー研究は独立課題として切り分ける。

## 1. L0：正しい機構か（完了）

ファイル `source/mechanism.py` / `tests/test_mechanism.py` / `results/RESULTS_L0.json`。未来トークン変化が過去出力に混入しないこと、同じQK回転がゲージであること、別回転でスコアが変わること、学習可能な角度への勾配、同一ReLU投影の冪等性、ビュー増加の実時間増大を検査済み。**学習成功・機能容量の証明ではない**。

## 2. L1：トークン間情報が本当に学習されるか（次に実施）

最初の対象は小型CPU、語彙64・系列長32・幅32・共有重み。タスクを分ける：
1. **Delay / Copy**：一定距離前の記号の復元。位置依存の情報アクセスが不可欠。局所混合または単純な時間ずらしで解ける場合はその通常対照を入れる。
2. **Content-addressed retrieval**：先行するキー・値の中から、クエリによる選択が必要な課題。単純なtoken-wise MLPでは解けない構成。
3. **Ordered transitions**：既知の規則列に従う状態更新・合成。順序・反復とMirror方向がどこに効くかを測る。
4. **Hidden-branch packet**（明示的負対照）：未来の分岐が現在のprefixでは決定不能。独立スロットで過剰な同時生成成功を主張しない。

開発世界候補618200、fresh候補618201～618203（**まだ生成・使用していない**）。2学習初期値/世界、更新数例1600、batch64、LR候補0.003/0.01。これらの数字はプロトコルの**草案**であり、ソース・対象分割・学習率選択を凍結してからfreshを開く。

### L1 比較法

- **Token-local control**：token-wise shared MLP + Mirror、系列情報なし（因果性の負対照）。
- **Native causal baseline**：幅・保存バイト・演算量を揃えた通常の小型Transformer。
- **Tied recurrent baseline**：同じブロックをT回使うUniversal Transformer様の制御。
- **Shared QKV**：Q/K/Vを共有しViewなし、または同一QK回転のみ（null）。
- **Native lightweight view**：gain/FiLM/IA3, rank-1/2, ordinary learned K head code.
- **Candidate**：異なる Q/K/View、非線形V Mirror、可変 K と反復 T、途中読み出し。
- **Untied K heads / T layers**：計算・容量の上界で、同バイトの採用比較ではない。

Mirrorが相対Q/Kを変えることで得られた品質を、単なる普通の相対QKアダプタが同バイトで再現する可能性を最優先で否定・確認する。

## 3. L2：KとTを独立に増やす

- K=1,2,4,8、T=1,2,4。固定 `K*T` と同一byte/FL0P両方を別集計。
- 同一入力のK方向について共有射影は1回。ただしK通りのAttention `O(K L^2 d)` が残る。
- 異なる反復tでは重みを共有するだけで射影行列積は繰り返す。コード数と演算回数を混同しない。
- Kを増やしてもAttention分布が同じ、利用頭が潰れる場合は表現増加なし。
- `d_virtual` と物理d、コードm、生成子psi、入力/output headを個別課金。付加価値は同バイトNLL/機能品質、学習コスト、active FLOPs、latencyで判断。

## 4. L3：出力する「時点」の制御

- 固定t早期readout、同時複数t readout、検証確信度/小さなhalting policy。
- 途中出力でも計算続行可。停止までのFLOPsとタスク損失のtrade-offで採用。
- 多未来トークンならjoint trajectory NLL・target p/q受理・rollback KVコピー・実測decode tok/s。単純なP個の独立正解率だけでは不可。

## 5. L4：小言語モデルへの移植（L1/L2成立後）

- 既存 `third_party/nanoGPT` を変更せず薄い実験モジュールで実装。
- ベースの標準Transformerと同データ/Tokenizer/Schedule/Budgetを使う。
- 自己回帰NLL、長系列/未知規則、prefill/decode/VRAM/KV、serialized byte、training FLOPsを比較。
- Transformer置換の主張はこの段階のfresh言語実験でのみ判断。

## 合否条件／停止条件

主ゲート：3 fresh worldsのうち少なくとも2世界で、候補が**最強の同バイトnative制御**に対しpaired sequence NLLを5%以上改善、worst-caseの劣化が10%を超えず、同時に学習コスト・推論コストも列挙。native方式が同等ならMirror固有採用はFAIL。これは**暫定の閾値案**で、L1開始前に固定すること。

特に、①未来情報漏洩、②実際には関数が同じQKゲージ、③K増分がほとんど死んだhead、④T増分が同一ReLUの冪等性だけ、⑤処理が遅すぎる、のいずれかなら、該当構造は失敗として保存する。

**他の研究を同時に増やさず、まず MA-1162 L1 token mixing に集中する。**
