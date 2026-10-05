# 到達コストを考慮したMirror解析 — RA-Mirror v1

日付: 2026-10-05  
状態: **解析的定式化と有限数値検算。新しいTransformer学習・実checkpoint再解析は未実施。**  
参照基点: `Unjuno/functional-quotient-compression@77e5b1dfefe7a08ee8b930cfe91e443ad45a75d8`。

## 0. 目的と結論

目的は「回転では表せない方向」だけでなく、**通常重みでも表せるが、指定した更新方法・既存能力保持条件・有限計算予算では到達しづらい機能を、Mirrorなら効率よく扱えるか**を分析すること。

本線はrouter-freeのまま保つ。入力ごとの学習router、state専用学習行列、正解を使う推論時state選択は追加しない。Mirror方向は開発データでオフラインに選び、学習・監査前に固定する。方向の生成に学習データを使ったなら、その探索費用と方向の記述量も数える。

三つを分ける。

1. **表現可能性**: 指定した基準モデルの局所接空間で同じ出力変化を実現できるか。
2. **到達効率**: 実現可能でも、所定の計量・保持条件・学習ステップ数では高コストか。
3. **課題価値**: その変化が実際に必要な改善か。大きく変わる・感度が高いだけでは有用とは言えない。

解析から得るのは条件付きの最小コスト、候補方向、局所次元、振幅の試行上限である。**課題・checkpoint・コスト定義なしの普遍的な最適Mirror、最適個数、容量倍率は導かない。**

### 以前の会話・解析メモへの訂正

| 以前の説明 | 正しい限定・修正 |
|---|---|
| baseline到達困難性を分母にした一般化固有値で、高コスト方向を選ぶ | 分母に置くと高コスト方向を避ける。到達コスト比ならbaselineコストを分子、Mirrorコストを分母にする。 |
| 勾配energyの大きい方向は有用 | それは一次感度。損失を悪化させる方向も含む。符号・曲率・既存能力・新しい開発データを確認する。 |
| 一つの固定Mirrorで通常FFNに表せない機能が増える | 可逆な線形共役なら通常のup/down重みへ厳密に畳み込める。単一viewの関数族拡大とは限らない。 |
| 共分散rankからK=5が最適 | K>=r+1は平均ゼロ・rank rの有限配置に関する下限。rの選択と容量最適Kは未確定。 |
| simplexならtoken一周期の一次作用が相殺 | tokenごとの感度が異なるため一般には相殺しない。ランダムな開始位相の期待値と一周期の実値は別。 |
| det(Q)=1なら安全 | 体積を保つだけ。特異値や条件数は任意に大きくなり得る。 |
| HessianとFisherは同じ損失曲率 | target NLLのHessianは不定にもなり得る。Fisher/GGNは別のPSD近似で、予測KL等の意味を明示する。 |
| directionをseedから読むので追加容量はほぼ無料 | データから求めたdense directionはseedだけでは復元できない。保存または有料の再生成手順が必要。 |

旧解析メモはRF1 Dense 6 checkpoint、16 dev例、局所勾配行列192個のスペクトルなどを報告していた。本更新でそのcheckpoint解析は再実行していない。旧メモのK=5/6/9等は**探索候補の履歴**であり、最適K、必要容量、再現済み新結果として昇格させない。原メモのSHA-256は検証フォルダの`provenance.json`に残す。

## 1. Mirrorの局所作用

### 1.1 変数表

以下の数値はモデル内部の無次元量。角度・振幅にも必要に応じ無次元の規格化を用いる。

| 記号 | 意味・定義 | SI単位 | 型・定義域・前提 |
|---|---|---|---|
| m,d | FFN中間幅、residual幅 | 1 | 正整数 |
| v,u | FFN中間入力、FFN入力 | 1 | 実ベクトル、R^m / R^d |
| phi,D_phi(v) | 成分別GELU、そのJacobian diag(phi'(v)) | 1 | 滑らかな関数 / m×m対角行列 |
| A,Q(s) | generator、exp(s A) | 1 | Aは実m×m行列、Qは可逆 |
| s | 一方向の局所変位 | 1 | 実スカラー、恒等近傍 |
| Psi(v;s A) | exp(-s A) phi(exp(s A)v) | 1 | R^m値関数 |
| Delta_A(v) | s=0におけるPsiの微分 | 1 | R^mベクトル |
| w,G(v,w) | Psi出力への損失勾配、Aへの行列勾配 | 1 | R^m / R^(m×m)、損失は規格化 |
| W_up,b_up,W_down,b_down | FFNの通常重みとbias | 1 | m×d,m,d×m,d |
| delta W_up,delta b_up,delta W_down | baselineで同じ一次作用を作る更新 | 1 | 対応する重みと同じ形状 |
| I,tr,det,kappa_2 | 単位行列、trace、行列式、2-norm条件数 | 1 | 文脈で次元を指定 |

### 1.2 一次作用の導出

```math
Q(s)=\exp(sA)=I+sA+O(s^2),\qquad Q(s)^{-1}=I-sA+O(s^2).
```

GELUのTaylor展開から、

```math
\phi(Q(s)v)=\phi(v)+sD_\phi(v)Av+O(s^2).
```

左から逆変換を掛けると、

```math
\Psi(v;sA)=\phi(v)+s\{D_\phi(v)Av-A\phi(v)\}+O(s^2).
```

したがって、

```math
\boxed{\Delta_A(v)=D_\phi(v)Av-A\phi(v).}
```

downstream勾配wに対する方向微分は、成分展開すると

```math
w^T\Delta_A(v)
=\sum_{ij}\{w_i\phi'(v_i)v_j-w_i\phi(v_j)\}A_{ij}.
```

よってFrobenius内積に関する行列勾配は

```math
\boxed{G(v,w)=(w\odot\phi'(v))v^T-w\phi(v)^T.}
```

これは一つの局所点の一次感度であり、平均勾配・非中心二次moment・中心化共分散は異なる量。共分散を報告する場合は中心化、標本単位、token/layerの集約方法を固定する。

### 1.3 回転以外の方向は残す。ただし局所energyで一律に選別しない

実行列Aは、skew-symmetric、symmetric-traceless、scalarの直交和へ分解できる。次元はそれぞれm(m-1)/2、m(m+1)/2-1、1。m=128なら8,128 / 8,255 / 1で、総数16,384。

skew成分の指数は回転、symmetric成分の指数は異方的伸縮を含む。単純shearのgeneratorは一般に非対称であり、両成分を持つ。単一の実matrix exponentialが任意の可逆行列を表すとは主張しない。

Frobenius直交分解は、課題上の直交分解ではない。ある勾配で非回転成分が大きくても、必要な能力の同じ割合が回転で不可能と結論しない。スカラー方向の小さい局所値から、すべての課題・学習時点で不要とも結論しない。

### 1.4 一固定viewは通常FFNへ畳み込める — 厳密な反例境界

一層で、全入力・tokenに同じ可逆Qを用いるとする。

```math
F_Q(u)=W_{down}Q^{-1}\phi\{Q(W_{up}u+b_{up})\}+b_{down}.
```

通常FFNの重みを `Q W_up`, `Q b_up`, `W_down Q^{-1}`, `b_down` とすれば、結合則によりすべてのuで同じ出力になる。biasも変換する必要がある。

Q(s)=exp(sA)をs=0で微分すると、baseline更新

```math
\delta W_{up}=A W_{up},\qquad
\delta b_{up}=A b_{up},\qquad
\delta W_{down}=-W_{down}A
```

が、同じMirror一次作用を実現する。したがって、全FFN重みを自由に更新できるbaselineに対して、一固定viewの方向を「構造的に到達不能」とするのは誤り。

それでも、固定optimizer、保持条件、量子化、疎な更新制約等の下ではコストが違い得る。そこが今回の対象である。

複数viewに一つの同じ更新を要求する場合や、tokenごとにQが変わる場合には、この単一の畳み込みを一般には使えない。ただし、入力と無関係なview差を作れること自体は課題容量増加の証明ではない。

**単位・形状確認:** Av、D_phi Av、A phi(v)はm次元。W_downを通すとd次元。損失勾配との内積は規格化した損失の一次変化となる。

## 2. 同じ機能変化に対するbaseline最小到達コスト

### 2.1 比較の単位を固定する

有限の開発witness上で、出力の定数shift等の不要自由度を除いた観測量を比較する。例えばcentered logitsを、固定された課題重みで白色化する。白色化・射影はcheckpointで凍結し、微分時に勝手に変えない。

| 記号 | 意味・定義 | SI単位 | 型・範囲・前提 |
|---|---|---|---|
| theta,delta theta | baselineの自由パラメータ、その更新 | 1 | R^n、自由にするparameter集合を宣言 |
| n,N,p | parameter数、観測次元、候補Mirror座標数 | 1 | 正整数 |
| f,J | 白色化した観測、d f/d theta | 1 | R^N、N×n |
| a,B | Mirror候補座標、d f/d a | 1 | R^p、N×p、a=0で線形化 |
| b | 比較する機能変化Ba | 1 | R^N |
| R_theta,R_M | baselineとMirrorの更新計量 | 1 | 対称正定値、n×n / p×p |
| W_theta | J R_theta^(-1) J^T | 1 | N×N、PSD、到達Gram行列 |
| C_base(b),C_M(a) | 規格化した更新費用 | 1 | 非負、baseline到達不能なら+infinity |
| Jbar,z,U,Sigma,V | whitened Jacobian、whitened更新、そのSVD | 1 | 下記定義・整合する行列サイズ |
| dagger | Moore–Penrose逆 | 1 | 明示するrank toleranceで数値近似 |
| r_null | baseline到達空間から外れた成分 | 1 | R^N |
| tau | 近似再現の残差重みパラメータ | 1 | 正スカラー |

R_thetaは単なる便利な正規化ではなく、比較の定義。例えばoptimizerの凍結preconditionerの逆、または宣言したparameter scaleに基づく計量を使う。R_Mにもgeneratorのscale、既存能力への作用を反映する。生の `||delta theta|| / ||a||` は座標のscale変更だけで任意に変わり得る。[S1,S2]

### 2.2 制約付き二次最小化と証明

```math
C_{base}(b)=\min_{J\delta\theta=b}\frac12\delta\theta^T R_\theta\delta\theta.
```

対称正定値平方根を使い、

```math
z=R_\theta^{1/2}\delta\theta,\qquad \bar J=J R_\theta^{-1/2}
```

と置くと、目的は||z||^2/2、制約はJbar z=bとなる。

Jbarのrankをrとし、正の特異値のみを並べたthin SVDをU Sigma V^Tとする。bがrange(Jbar)に属するとき、制約を満たす全解は

```math
z=V\Sigma^{-1}U^Tb+z_{null},\qquad \bar Jz_{null}=0.
```

二項は直交するので、ノルム二乗は和になる。最小解はz_null=0。よって、

```math
\boxed{
\delta\theta_* = R_\theta^{-1}J^T W_\theta^\dagger b,
\qquad
C_{base}(b)=\frac12 b^T W_\theta^\dagger b.
}
```

ただし、この式は**到達可能性の確認後だけ**使う。

```math
r_{null}=(I-W_\theta W_\theta^\dagger)b.
```

r_nullが非ゼロなら等式制約は実現不能で、厳密コストは+infinity。pseudoinverseはnull方向へ0を返すため、これを先に確認しないと「到達不能だから無料」という逆の誤りになる。

実モデルでは有限witness・数値rankに関する判定であり、全データ上の表現不能定理ではない。

### 2.3 近似再現は別の問題

完全な再現ではなく、

```math
\min_{\delta\theta}
\frac12\delta\theta^T R_\theta\delta\theta
+\frac1{2\tau}\|J\delta\theta-b\|^2
```

を解くなら、stationarityから

```math
(R_\theta+J^TJ/\tau)\delta\theta=J^Tb/\tau,
```

したがって

```math
\delta\theta_\tau=R_\theta^{-1}J^T(W_\theta+\tau I)^{-1}b.
```

両辺へ前式の係数行列を掛ければ等式を確認できる。最小の**罰則込み目的値**は b^T(W_theta+tau I)^(-1)b/2。ただし、実際の更新費用と残差費用は別に報告する。ridgeで有限値が出たことを完全到達の証明にはしない。

### 2.4 既存能力への干渉

新課題と旧課題をstackして目標を「新課題の変化、旧課題は0」とする方法と、旧課題の作用を二次罰則へ入れる方法を分ける。

追加記号: `J_old`は旧課題観測のbaseline Jacobian、`B_old`はMirror Jacobian、`mu>=0`は規格化した干渉罰則、`R_theta,0`と`R_M,0`は元のSPD計量。すべて無次元の行列・スカラー。

```math
R_\theta=R_{\theta,0}+\mu J_{old}^T J_{old},\qquad
R_M=R_{M,0}+\mu B_{old}^T B_{old}.
```

**Mirror側も旧課題を壊し得る。** baselineだけへ罰則を課してMirrorを有利にしない。旧課題0というhard constraintを使う場合はMirror側も同条件を満たす必要がある。

wall-clock秒、serialized bytes、規格化した更新ノルムは単位も意味も違う。任意に足した量を物理的なコストと呼ばず、原則として制約付きPareto比較にする。

### 2.5 座標変更に対する確認

可逆な座標変更delta theta=T delta betaを使うならJはJT、計量はT^T R_theta Tへ変える。このとき

```math
(JT)(T^T R_\theta T)^{-1}(JT)^T=J R_\theta^{-1}J^T.
```

右辺と左辺は一致するので、上の最小コストは不変。計量を変えずにparameterの単位だけ変えた比較は、別のoptimizer・費用を比較している。

## 3. 正しい効率固有値問題と、課題有用性の分離

### 3.1 「同じ変化を安く出す」ための比

Bの像をbaseline到達可能部分へ制限する。次を定義する。

| 記号 | 意味 | SI単位 | 型・条件 |
|---|---|---|---|
| R_B | B^T W_theta^dagger B | 1 | p×p、PSD、baseline模倣コスト行列 |
| E(a) | C_base(Ba)/C_M(a) | 1 | a!=0での費用比 |
| v_i,lambda_i | 一般化固有vectorと固有値 | 1 | R_M正規化、lambda_i>=0 |
| H | R_M^(-1/2) R_B R_M^(-1/2) | 1 | 対称PSD行列 |
| g_f,beta | 観測への損失勾配、B^T g_f | 1 | R^N、R^p |
| H_a | Mirror座標に対する実損失Hessian | 1 | 対称、PSDとは限らない |

```math
C_M(a)=\frac12a^TR_Ma,\qquad
\boxed{E(a)=\frac{a^TR_Ba}{a^TR_Ma}.}
```

a=R_M^(-1/2)zに置換すると、Eはz^T H z/(z^T z)。Hを直交対角化すれば、この比は固有値の非負重み付き平均なので、最大値は最大固有値。その方向を戻すと、

```math
\boxed{R_Bv_i=\lambda_i R_Mv_i.}
```

大きいlambdaは、**その出力変化をbaseline重みで模倣する費用がMirror計量に比べて大きい**ことを表す。課題有用性、学習成功、時間短縮、容量倍率は表さない。

以前の `G v=lambda R v` でRをbaseline困難性と説明した式は、この目的に対して逆向きだった。また「utility × baseline cost / Mirror cost」は一般にRayleigh商ではなく、そのまま同じ一般化固有値問題へ還元できない。

### 3.2 数値例

```math
J=\operatorname{diag}(1,0.1),\quad
R_\theta=R_M=B=I.
```

W_theta=diag(1,0.01)、R_B=diag(1,100)。二方向の同じ大きさの出力変化に、baselineでは0.5と50の費用が必要。Mirror計量では双方0.5。よって費用比は1と100。

逆向きの `I v=lambda R_B v` の最大固有値は簡単な第一方向を選んでしまう。今回の数値検算は両方を確認する。

### 3.3 有用性を別に確認する

候補aへの小さい変位sに対して、

```math
L(sa)-L(0)=s\,\beta^Ta+\frac{s^2}{2}a^TH_a a+O(s^3).
```

費用比が高くても、課題と無関係、悪化方向、または旧課題損傷が大きければ不採用。符号を選べる一方向probeならbetaとの内積と曲率を評価する。**router-freeの平均ゼロview訓練では両符号を使うため、一方向の下降可能性をそのまま平均学習利得へ変換しない。**

候補の採否は、課題改善・既存能力保持・記述量・実行量の制約付き問題。制約を追加した問題の厳密解が、無制約の固有vectorと同じとは限らない。

## 4. 「可能だが学習では遠い」を有限ステップで測る

追加記号: `b`は目標出力差、`delta theta_j`はj更新後の重み差、`e_j=b-J delta theta_j`は残差、`eta`は固定学習率、`T`は非負整数の更新回数。すべて無次元、Tは回数。固定Jと固定R_thetaによる二乗誤差だけを仮定する。

```math
\delta\theta_{j+1}=\delta\theta_j+\eta R_\theta^{-1}J^Te_j.
```

Jを掛けて目標から引くと、

```math
e_{j+1}=(I-\eta W_\theta)e_j.
```

delta theta_0=0から帰納的に、

```math
\boxed{e_T=(I-\eta W_\theta)^T b.}
```

最後のTは**転置でなく整数乗**。固有値omegaの成分は(1-eta omega)^T倍になる。0<eta<2/lambda_max(W_theta)なら正固有値方向は収束し、null方向は残る。小さい正固有値は到達可能でも有限時間では遅い。

これは固定線形二乗問題の厳密式。実際のAdamW、変化するJacobian、非凸障壁を厳密に予測するものではない。[S1] 実モデルでは10/50/100等の短いstep数を事前固定してreplayし、予測と実測の差を記録する。

### router-free学習へ戻すための必須条件

本番で学習するのはthetaであってaではない。aだけを最適化したprobeは座標の診断に過ぎない。

採用候補を固定viewへ変換した後、**全viewの出力とtheta-Jacobianを、view確率の平方根で重み付けしてstack**する。同じ入力が複数viewにあっても教師は同じ。以後は上式をそのstacked Jと目標へ適用して、実際に共同更新されるthetaの局所学習作用を比較する。各view単独のJacobianだけで、多view共同学習の収束を保証しない。

追加の学習router、view別に自由な更新方向、oracle課題分岐を暗黙に使うprobeは本線の証拠にしない。

## 5. Mirror個数Kについて解析で言えること

### 5.1 最小supportの定理

| 記号 | 意味 | SI単位 | 型・前提 |
|---|---|---|---|
| r | 採用した候補部分空間の次元 | 1 | 正整数、事前診断・開発検証で選ぶ |
| K | finite view数 | 1 | 正整数 |
| c_k,pi_k | r次元codeとその確率 | 1 | c_k in R^r、pi_k>0、sum pi_k=1 |
| C,V | codeを行に並べたK×r行列、その構成用基底 | 1 | V^T V=I、V^T 1=0 |
| Sigma_c | sum pi_k c_k c_k^T | 1 | 平均ゼロ時の共分散 |

平均ゼロ条件はC^T pi=0。piは非ゼロvectorなのでrank(C)<=K-1。Sigma_cのrankは、正の確率の下でrank(C)と等しい。したがってrank rを要求すると、

```math
\boxed{K\ge r+1.}
```

K=r+1とし、1の直交補空間の正規直交基底Vを取り、

```math
C=\sqrt{K/r}\,V
```

と定義する。V V^T=I-11^T/Kより、

```math
CC^T=(K/r)I-(1/r)11^T.
```

従って各行のノルムは1、異なる行の内積は-1/r、平均は0。一様確率で、

```math
C^TC/K=I/r.
```

これが正則simplexによる下限達成である。費用計量R_Mが単位行列でない場合は、採用部分空間をその計量で白色化してから配置し、元の座標へ戻す。

### 5.2 この定理が答えないこと

K=r+1は、**指定したr次元の平均・共分散を実現する最少点数**であり、次の最適解ではない。

- 容量最大、学習最速、汎化最良。
- 許容領域全体を覆う最少点数。
- 周期実行で最も有利なstate順序。
- 非対称な損失や高次相互作用に最適な配置。

従って旧メモのK=5/6/9等は候補として残せるが、今回の解析だけから最適値を確定しない。rを決める「energy 90%」も、安全性や課題保持の定理ではない。費用比だけ大きく課題価値のない方向もあるため、有用性を確認した部分空間の追加利得でrを選ぶ。

## 6. token周期では、平均ゼロだけで相殺しない

追加記号: `t`は論理token位置、`o`は系列開始位相、`k(t,o)`は実行state、`beta_t`はtoken位置の方向感度、`H_tu`はtoken位置間のHessian block。いずれも無次元、t/o/kは整数。teacher-forced入力を固定し、全因果経路を微分する。

```math
k(t,o)=(t+o)\bmod K.
```

一周期の一次変化は、一般に

```math
\sum_t\beta_t^T c_{k(t,o)}
```

である。sum c_k=0でもbeta_tはtokenごとに違う。最小反例はcode=(1,-1)、感度=(1,2)で、code和0に対して機能的な一次和は-1。

開始位相oを全K通りで一様に選び、同じ入力・基点で比較するなら、各固定tでE_o[c_k(t,o)]=0となり、**期待値**の一次項は0になる。ただし、二次項には

```math
\frac{\rho^2}{2}\sum_{t,u}
\mathbb E_o[c_{k(t,o)}^T H_{tu}c_{k(u,o)}]
```

が含まれる。周期コードはtoken間で相関するため、t!=uの項を落とせない。

同じことが層間にも当てはまる。各token/layerの行列勾配を別標本としてPCAするだけでは、共同介入の損失曲率を再現しない。局所energyの和と、共同方向微分の二乗は異なる。

**実装条件:** global state、開始位相、論理token offset、padding規則、document reset、BOS扱い、cache offsetを固定する。chunk分割によってphaseをリセットしない。周期的FFN切替はtoken自体の順序を混ぜる操作ではなく、位置に応じた演算子切替である。語の役割とphaseが偶然固定対応していないか、prefix長・開始位相を変える対照を置く。

## 7. 振幅rho: 曲率は試行値の計算に使い、保証とは区別する

### 7.1 変数表

| 記号 | 意味 | SI単位 | 型・前提 |
|---|---|---|---|
| rho | Mirror振幅 | 1 | 非負、generatorの規格化とセット |
| a_k | 同じ基点の全介入vector | 1 | token/layerをstackしたvector、平均0 |
| H_L,Sigma | 真の損失Hessian、介入共分散 | 1 | 対称 / PSD |
| hbar | tr(H_L Sigma) | 1 | 符号不定 |
| M3 | 領域内の3次微分の作用素ノルム上限 | 1 | 非負、既知の場合のみ保証に使用 |
| epsilon | 許容する平均損失悪化 | nat/教師token、SIでは1 | 正スカラー |
| amax,kappa_max | 最大generator 2-norm、許容条件数 | 1 | amax>=0、kappa_max>=1 |

### 7.2 平均損失と剰余

同じ入力、同じtheta、平均ゼロの全介入でTaylor展開すると、

```math
\mathbb E_k[L(\rho a_k)-L(0)]
=\frac{\rho^2}{2}\operatorname{tr}(H_L\Sigma)+R_3.
```

3次微分の上限M3が有効なら、

```math
|R_3|\le\frac{M_3\rho^3}{6}\mathbb E_k\|a_k\|^3.
```

従って、hbar>0で高次項を無視できるときの**試行値**は、

```math
\rho_{trial}=\sqrt{2\epsilon/\bar h}.
```

hbar<=0、推定が不安定、領域外に出る場合には、この平方根式は採用できない。保証付き上限が必要なら、二次項と上の剰余上限の和がepsilon以下になるrhoを解く。M3が未測定なら、開発データの有限forward replayで振幅を縮小して検証する。target NLLの実Hessianと、予測KLを測るFisher/GGNは区別する。[S1]

### 7.3 行列の安定性

tr(A)=0ならdet exp(rho A)=exp(rho tr A)=1。ただしA=diag(3,-3)ならrho=1で条件数はexp(6)≈403.43。体積保存は安全性ではない。

matrix exponentialの級数と劣乗法性から、

```math
\|e^{\rho A}\|_2\le e^{\rho\|A\|_2},\qquad
\|e^{-\rho A}\|_2\le e^{\rho\|A\|_2},\qquad
\kappa_2(e^{\rho A})\le e^{2\rho\|A\|_2}.
```

したがって、amax>0ならrho<=log(kappa_max)/(2 amax)を十分条件の一つとして使える。skew方向には保守的なboundであり、全方向の正確な条件数ではない。実装では条件数だけでなくQ・Q逆の個別norm、activation、FP32/BF16誤差も測る。

## 8. 到達効率と保存容量の混同を避ける

データから求めたAを固定しても、その行列は記述すべき学習成果物。gradientで更新しないから無料にはならない。

例えば各層で4本のdense 128×128 generatorを12層分保存すると、786,432 scalar、FP32で3,145,728 byte（3 MiB）。これは従来RF1の438,912学習scalarのFP32 payload 1,755,648 byteを上回る。metadata、コード、共通runtime、生成コストはさらに別。

したがって候補には、固定されたdecoder既知basis、疎なgenerator、block構造、低rankな近似等を用意し、**近似後の機能差と到達効率**を再評価する。全行列を保存した上でseedだけのモデルサイズとして報告しない。

同様に、固定Mirrorの座標を変えるprobeで速く到達できても、router-free訓練のtheta更新が速いとは限らない。採用判定は実serialized bytes、実training work、one-view品質で行う。

## 9. 次の解析・実験プロトコル

### H: 仮説

課題に有用で、既存能力を保持し、通常更新では高コストな方向を選ぶMirrorは、単なる勾配energy上位・ランダム回転より良い有限予算の学習/容量効率を生む。**本更新はその仮説の証明ではない。**

### T: 最小手順

1. 既存のRF1 source/checkpoint・データをhash固定。optimizer、許すbaseline更新集合、観測の白色化、既存課題保持条件を明記する。
2. 開発データを、方向提案・幾何推定・候補選択の非重複部分へ分ける。内容単位を独立単位とし、同じ内容のtoken/layerを独立標本扱いしない。
3. 候補generator族を宣言。回転、symmetric-traceless、scalar control、低記述量の混合を含め、seedと初期学習状態を都合よく選ばない。
4. JVP/VJPでJとBの作用を計算。大きいJacobian全体を保存することは必須でない。小fixtureでdense列挙と一致を検証する。
5. baseline可到達性・最小更新費用・旧課題への作用を計算。metric、rank tolerance、ridge、観測重みへの感度を報告する。
6. 費用比の固有方向を提案し、別開発部分で符号付き課題変化と有限振幅を検証する。共分散energyだけで採用しない。
7. 記述予算に合う候補部分空間を選ぶ。rごとにsimplex r+1、antithetic 2r等を比較。下限達成と最適Kを同一視しない。
8. token/layer scheduleを含む共同Hessianと開始位相を使ってrhoの試行値を決める。正規化、condition bound、有限forwardで検証する。
9. 同じ親checkpointから、通常更新・固定Mirrorありのtheta更新・等費用の別preconditioner/正則化を比較する。Mirror座標だけを最適化する試験は補助診断に分離する。
10. 最後に新しいworld、model seed、Mirror seedで設定を凍結して確認。one-view能力、独立規則保持、保存量・計算量を評価する。

proposal/selectionにはauditを使わない。従来RFの個数・変形幅の暫定値を最適解として固定しない。実測がないため、本書では新たな最適r/K/rhoの数値を宣言しない。

### D: 採否

- **PASS (mechanism)**: 局所予測が別開発データと有限step replayでも機能改善・保持条件を満たし、実費用で有利。
- **PASS (capacity)**: さらに固定全保存量・学習計算予算で、強い対照より多くの共有/固有規則をone-viewで保持し、新worldで確認。
- **FAIL**: 事前に固定した改善幅へ届かないこと、または品質・費用条件の違反が十分な精度で確認される。
- **UNCERTAIN**: 推定区間が閾値をまたぐ、rankやmetricの選択で結論が反転する、非線形近似が破れる。

本書では代数fixture以外の新しい標本がないため、実モデルのn_min/最小検出差は未確定。実checkpoint解析の実行仕様を凍結する段階で既存pilotの分散と予算から設定する。有限代数検算に統計的PASSを付けない。

### C: 主な破れ方

座標scaleの見かけの利益、Mirrorにも同程度の旧課題損傷、baselineの許す更新集合が不当に狭い、追加教師情報やstateへのoracle割当、seed過適合、step数とFLOPsの混同、周期と課題token位置の相関、dense generatorの保存費用の未計上。

### U: 不確かさ

有限witnessによるrank、Hessian/GGN差、optimizerの時間変化、非線形Taylor剰余、code/basis選択、FP丸め、有限の世界・学習seed。モデル品質の合成標準不確かさu_cと包含係数は未推定。実測誤差と証明上の仮定を混同しない。

## 10. 今回実行した検算

[有限検算コード](../../experiments/analytic_mirror/reachability_v1/local_geometry.py)、[15件の回帰テスト](../../experiments/analytic_mirror/reachability_v1/test_geometry.py)、[生結果](../../experiments/analytic_mirror/reachability_v1/verification.json)を保存する。

再実行:

```bash
cd experiments/analytic_mirror/reachability_v1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python test_geometry.py
```

Python 3.13.5 / NumPy 2.3.5 / SciPy 1.17.0 / float64の有限小行列で15件通過。性能ベンチマークではない。最初は未実装の12検査が失敗し、実装後に全15件が通過した。

- 非回転を含むgeneratorの有限差分と一次式。
- downstream勾配とgenerator行列勾配の内積一致。
- 一固定viewのDense畳み込み・baseline接空間による模倣。
- SPD計量付き最小ノルム・到達不能方向の検出。
- 正しい/逆向き固有値問題の反例。
- 座標変更に対する計量の共変性。
- baselineとMirror双方への旧課題罰則。
- 固定線形系の20-step残差。
- r=1,2,4,8のsimplexの平均・共分散・Gram行列。
- token周期の一次相殺に対する反例とcross-token二次項。
- det=1でも条件数が大きい例。
- 平均ゼロ介入の二次Taylor式と入力検査。

実測例: 一次式の中央差分最大誤差3.27e-12、固定view畳み込み差2.11e-15、最小コスト等式制約残差2.99e-16。いずれもこのfixtureの結果で、実checkpointの学習成功や大規模数値安定性の保証ではない。

## 11. 根拠・関連研究

- [P1] repository: [T153–T161 Mirror診断](../MIRROR_HYPOTHESIS_T153_T161.md)。rankだけでなく課題固有差・推定誤差を区別する既存の限定された合成系の証拠。
- [P2] repository: [MN010までの状態](RESEARCH_STATE_THROUGH_MN010.md)。Mirror個数、機能breadth、Dense幅の区別。
- [S1] James Martens, *New Insights and Perspectives on the Natural Gradient Method*, JMLR 21(146), 2020. https://jmlr.org/papers/v21/17-678.html ; arXiv:1412.1193. Fisher/GGN、計量、damping、二次法の背景。
- [S2] Laurent Dinh et al., *Sharp Minima Can Generalize For Deep Nets*, ICML/PMLR 70, 2017. https://proceedings.mlr.press/v70/dinh17b.html ; arXiv:1703.04933. 生のparameter-space geometryを一般化と直結させない根拠。
- [S3] Maksym Andriushchenko et al., *A Modern Look at the Relationship between Sharpness and Generalization*, ICML/PMLR 202, 2023. https://proceedings.mlr.press/v202/andriushchenko23a.html . flatnessを汎化保証として扱わない背景。

本書の最小コスト、Rayleigh商、simplexのrank下限は上記仮定から本文中で導出した標準的な線形代数・局所最適化。これらを新規な数学定理や、既報論文がMirrorの性能を保証したものとして主張しない。

## ERROR CHECK

固定viewの表現可能性、費用比の向き、到達不能の扱い、課題有用性、token相互作用、最小supportと最適K、Hessian/Fisher、行列安定性、direction保存費用を分離した。実checkpointの新しい学習・到達費用・最適K/rhoは**未測定**。本更新は前の定式化の訂正と、次に測る量の確定である。
