# MA-691 次段階実験計画 — Mirror共通KVキャッシュの性能と品質を反証可能に検証

**作成日：2026-10-18 / 対象ブランチ：research/ma-691-cache-aware-decode-plan-20261018**

## 0. 結論と優先順位

最優先は**共通正準K/Vキャッシュを、MirrorごとのQuery/出力変換で直接参照する方式**。二番目に構造化Mirrorカーネル融合、三番目に仮想重みの圧縮。巨大高精度1スカラー方式は独立研究として保持するが、このKV実験の採否に混ぜない。

既存のCPU単一トークン演算診断（AMD EPYC仮想CPU、Torch FP32、1スレッド、H=4、d=32、prefix=8192、K=8）では、正準K/V約8MiB・8組の展開K/V約64MiB。ある測定では正準約2.50ms対展開約3.97msだった。ただし別の同型再実行では正準2.69ms対展開3.76ms。**この速度は実LLMのtok/sではない。** 128〜2048トークンでは正準が遅い場合があり、Q-only型の既存共有方式はさらに速い場合もあったが、そこではView出力の機能が等価ではない。

**優先判断は暫定的**。次の各ゲートを突破するまで「Mirror固有のLLM高速化」や「通常GQAより優秀」とは記載しない。

## 1. 厳密な数理条件

1つのproducerから作った同一履歴のK/Vを共有する。論理Viewごとの変換を固定された右作用と仮定する。

\[
K_j=K_0 A_j,\qquad V_j=V_0 B_j
\]

\[
\begin{aligned}
\operatorname{Attn}(q_j,K_j,V_j)
&=\operatorname{softmax}\left(\frac{q_j (K_0 A_j)^\top}{\sqrt{d}}\right)(V_0 B_j)\\
&=\left[\operatorname{softmax}\left(\frac{(q_j A_j^\top)K_0^\top}{\sqrt{d}}\right)V_0\right]B_j .
\end{aligned}
\]

この変形にはA/Bの直交性は不要だが、形状と固定された右作用、および各Viewの履歴元状態が共通であることが必要である。RoPEを前段で適用する場合、変換順序の可換性または位置依存の正しい再配置が必要。

**特に重要な非Mirror対照：** 固定されたA/Bなら、Query側変換や出力線形射影へ作用を折り込めることがある。この「代数的に吸収した通常実装」は、元のMaterialized Viewに対しても**完全に機能等価**な直接対照とする。これより高速・小容量、または同じ予算で高品質でなければ、Mirror固有の採用理由は弱い。

\[
M_{\mathrm{canonical}}=2BHTdb,\qquad
M_{\mathrm{materialized}}=K M_{\mathrm{canonical}}
\]

ここではコード、索引、アラインメント、allocator断片化、一時workspaceは別加算。例：B=1,H=4,T=8192,d=32,b=4 byte、正準8MiB、K=8で全展開64MiB。64MiBに「8モデル別々のパラメータ」は含まれない。

### 変数表

| 記号 | 意味（日本語） | SI単位 | 定義・範囲・前提 | 型 |
|---|---|---|---|---|
| \(q_j\) | View jの現在トークンQuery | 1 | 1×d | 行ベクトル |
| \(K_0,V_0\) | 共通履歴の正準キー・値 | 1 | T×d（B,Hは外側に配置） | 行列 |
| \(A_j,B_j\) | キー・値を規定するView右作用 | 1 | d×d、既知、正しい順序 | 行列 |
| \(K_j,V_j\) | View jの論理キー・値 | 1 | T×d | 行列 |
| \(j,K\) | View番号と論理View総数 | 1 | 1≤j≤K、正整数 | 整数 |
| \(B\) | バッチ数 | 1 | 1,4等 | 整数 |
| \(H\) | 物理KVヘッド数 | 1 | 正整数、当初4 | 整数 |
| \(T\) | 履歴トークン長 | 1 | 128〜8192（拡張は別ラベル） | 整数 |
| \(d\) | 1ヘッド次元 | 1 | 正整数、当初32 | 整数 |
| \(b\) | 1要素の保存バイト数 | byte | FP32は4 byte | スカラー |
| \(M_{\mathrm{canonical}}\) | 正準K/Vの理論payload容量 | byte | 非負 | スカラー |
| \(M_{\mathrm{materialized}}\) | K組の全展開payload容量 | byte | 非負 | スカラー |

**次元チェック：** \((1\times d)(d\times d)(d\times T)\) は \(1\times T\)、これと \(T\times d\) の積は \(1\times d\)。容量式は要素数（無次元）×byteなのでbyte。速度の実測値はmsで記載し、転送バイトの削減だけから演算速度を推定しない。

## 2. 段階的な検証

| 段階 | 作業・介入 | 必須対照 | 進行判定 |
|---|---|---|---|
| L0a 厳密性 | 既存MA-691コードの再実行、RoPE整合性、変換順序、キャッシュproducer/version/トークン列の証明 | 明示的K/V変換・共通QKゲージ・非可換RoPE・MA-253中間層反例 | FP32差<=2e-6、invalid cache reuseをすべて拒否 |
| L0b メモリ | 物理ポインタ共有、allocator実容量、コード・索引・workspace、K=1/2/4/8 | 正準・K組独立・静的吸収型GQA | Kを増やしても実K/V payloadは1組、metadata別課金 |
| L0c 実行 | 実際に256トークンを自己回帰生成するprefill＋decodeのend-to-end時間、切替dwell | 全展開、毎切替再生成、吸収済み通常共有KV、Q-only非等価対照 | 比較母集団を維持してp50/p90・tok/s・TTFTを測る |
| L1 学習品質 | 小因果モデルを同じデータで学習。Q/Vコードを変えた場合の機能差と自然文NLL | MHA品質上限、GQA/MQA、MLA、通常Q/outputアダプタ | NLL・コピー/検索の品質を保持したPareto改善のみ主張 |
| L2 カーネル | CPUのNumba融合と最適化された通常実装、利用可能なら同条件のGPU Flash/PagedAttention系 | 同一thread/precision/cache layoutのnative kernel | コンパイル時間込みと定常時を分離し、実GPU差だけをGPU結果とする |

**順序：L0a→L0b→L0c→L1→L2。** L0で失敗した特性をL1で説明せず、先にバグか数理境界かを分類。L1自然文はデータ/Tokenizer/モデルrevの固定後だけfresh実行する。L2はGPU不在ならCPUまでを完了とし、ユーザーローカルへの作業移管はしない。

## 3. Frozen L0 実験行列と方法

詳細機械可読値は \`PROTOCOL_L0_FROZEN.json\`。記述上の標準条件：

- データ：決定的CPU乱数で生成した**モデル学習ではない**Key、Value、Query。開発691200、新しい数理fresh 691201/691202/691203。既存MA-691の開封済みseed69101〜69103は比較史としてのみ参照。
- サイズ：head数4、head幅32、FP32、KV長128/512/2048/8192、同時View数1/2/4/8。Batch1を全実測、Batch4はメモリ安全ゲート下の二次実測。
- 性能：同じQuery/Prefixに対し256トークン生成、View切替周期1/8/64/256。ウォームアップ10、各測定31回、ペア比較の実行順をランダム化。CPU PyTorchスレッド1、クロック未固定を報告。
- 特に「8方向を全て実行」・「一つだけ選択」・「切替時に再生成」は計算要求が異なるため別テーブル。各Viewの注意演算はKに応じて残り、記憶容量削減を計算省略として計上しない。
- すべての出力・ロジット誤差、cache pointer/storage ID、実割当バイト、CPU RSS peak、View metadata/索引バイト、prefill/TTFT、switch、tok/sをCSV化。未解決のメモリ超過条件は事前閾値超過としてSKIPPEDを記録する。
- **反例：** 中間層のMirrorでhiddenが変わる場合、後続KVキャッシュは別。異なるモデル・adapter・token prefixの無条件aliasを拒否。RoPEと非可換な順序を同一だと誤判定しない。
- **構造的な追加点：** static Viewは吸収可能なnative対照で確認し、tokenごとにMirrorプログラムが切り替わるdynamic Viewでは、その選択コード・switch latency・routing/条件入力も課金。

## 4. L1 ドラフト（学習済みモデルに拡張）

学習データのクラスは、(i) causal Copy/Content Retrieval/Order依存 synthetic、(ii)独立文書分割した実言語コーパス。実言語側はデータURLとimmutable revision、Tokenizer hashと重複しないdocument IDsを確定するまでL1を**凍結しない**。過去のL0乱数結果からL1の学習率を調整しない。

品質比較は以下を最優先にする：

1. native MQA/GQA（実際に1〜少数の物理KVヘッドを共有）とMLA、同一学習予算/保存バイトの通常Q/outputアダプタ。
2. Mirror付き共通KV（静的Givensで開始、DEVのみHouseholder/Butterflyへ拡張）。
3. materialized View K/V（Mirror候補と関数等価な参照）と無制限独立ヘッド（品質上限）。
4. adapter switching用途へ進む場合はLRAgent/PReCache/aLoRAを追加。独立学習済みadapterの後段隠れ状態が一致しないことを最初から明示する。

開発のみでLR・hidden幅・K/コード数・View配置・生成スケジュールを選ぶ。最低3 fresh独立世界×2初期seedで評価し、評価指標を「validation NLL（nat/token）」「Copy/Retrieval精度」「キャッシュalias/容量」「pre-fillとdecodeの総tokens/sec」「View更新とrouter bytes」「学習FLOP/時間」に分ける。固定更新だけでcapacityと呼ばない。

### 採用条件（L1凍結時までに最終固定する暫定ゲート）

質量・演算をそろえた**最強の通常KV共有方式**に対し、少なくとも2/3 fresh世界で以下のどちらかを満たすこと：

- **速度優位：** test NLL悪化が相対1%以内で、全prefix処理込みend-to-end生成tokens/secが10%以上改善、メモリ容量が悪化しない。
- **機能優位：** end-to-end遅延増加10%以内・実payload容量以下で、test NLLが相対2%以上改善する。

なお数学的に等価な静的folded nativeより遅ければ**Mirror固有の速度優位はFAIL**とする。新技術を使ったというだけで採用しない。

## 5. H / T / D / C / U

**H — 反証可能仮説：** cache-compatibleなViewプログラムに限定すれば、1組の正準KVでK個の異なるAttentionを計算でき、頻繁な切替や長文脈の条件で、通常KV共有実装と比べて品質・メモリ・速度のParetoフロンティアを改善し得る。

**T — 最小検証：** L0で新規数値seed3件×固定の形状行列、CPU1スレッド、同じ入力に対して等価Reference/absorbed nativeを並走。L1では3独立freshデータ世界×2seedを最小、反復計時31回、出来るだけ逐次停止なし。厳密性が落ちたときのみ構造判定を停止し、そのFAILを保存。

**D — PASS/FAIL/UNCERTAIN：** L0 algebra PASSはFP32等価差<=2e-6、invalid aliasの全拒否、実共有ストレージ。GPU不在や自然文未学習時は品質・GPU速度についてUNCERTAIN。L1 Mirror-specific PASSは上記Paretoゲート。簡単なnative Query/output吸収に同等以下ならM0/FAIL。

**C — 対立仮説：** 既存のGQA/MQA/MLAがほぼ同じキャッシュ共有を達成し、固定Mirror変換はQuery/出力への吸収と等価。長文脈で観測した利得は「非効率な多重materialize対照」に固有で、Mirror自体にはない。

**U — 不確かさ：** CPU仮想化、クロック変動、Torch eagerと融合kernel差、前段pre-fill込みか否か、RoPEとadapterのcache provenance、ヘッド数/文脈長、自然文データ由来の誤差。測定ごと中央値・p90・ペア差の95% bootstrap CIを報告。校正済み合成標準不確かさ \(u_c\) は得られない場合「未推定」、包含係数 \(k\) も未指定と明記する。3世界だけでは厳密な効果一般化の証明には不十分。

## 6. 予定する成果物とGit/研究規律

L0実行前に \`source/kv_decode.py\`, \`tests/test_exact.py\`, \`tests/test_cache_provenance.py\`, \`tests/test_rope.py\` を実装、仕様→テスト→事前SHA-256→freshの順で実行する。終了後 \`results/RESULTS_L0.csv\`, \`results/RESULTS_SWITCH.csv\`, \`results/RESULTS_MEMORY.csv\`, \`VERIFICATION_L0.json\`, \`REPORT_L0_JA.md\` にすべての未達と省略を記録する。

L1はソース/データ/Tokenizer hashとfresh分割を記した\`PROTOCOL_L1_FROZEN.json\`を別コミットに置いた後だけ開始。性能比較はCPU本体で完結させ、GPUが利用できないならGPU適用を装わない。

GitHub：既存MA-691フォルダ配下の**続編のみ**を更新する。新規MA番号の確保・\`main\`変更・過去MA-691 PROMISINGやMA-253 FAILを上書きしない。非再現・速度退行もCSVに残し、GitHub push後に生ファイルとコミットSHAを読み戻す。

### 優先する一次研究

- Ainslie et al., GQA, EMNLP 2023, DOI: 10.18653/v1/2023.emnlp-main.298
- Kwon et al., PagedAttention, SOSP 2023, arXiv:2309.06180
- Dao, FlashAttention-2, arXiv:2307.08691
- DeepSeek-AI, MLA (DeepSeek-V2), arXiv:2405.04434
- Jeon et al., LRAgent, ICML 2026, https://proceedings.mlr.press/v306/jeon26b.html

## ERROR CHECK（計画段階）

- 数式は行列次元と単位の整合性を確認。
- L0は固定プロトコル登録のみで、**fresh結果はまだない**。
- L1はDRAFTであり、まだモデル品質や実tok/sを主張しない。
- 既存のMA-691, MA-253, MA-1164成果と否定結果は継承し、上書きしない。

## 50字超要約

共通KV＋Mirror読み出しは長文脈と多方向切替で有望な初期機構だが、既存GQA/MQA/MLAに勝った証拠はない。次は厳密な再利用条件、実ポインタ共有、キャッシュを含めた全トークン生成速度、学習した自然文モデルの品質を別々に測定し、代数的に吸収したnative方式を最強対照としてParetoゲートを判定する。
