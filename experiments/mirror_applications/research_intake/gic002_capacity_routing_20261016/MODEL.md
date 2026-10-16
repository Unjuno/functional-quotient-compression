# GIC002 mathematical contract

## Variable table
|Symbol|Meaning (日本語)|SI unit|Definition|Domain/assumptions|Type|
|---|---|---|---|---|---|
|d,h,q|入力幅・隠れ幅・回転方向数|1|5,16,q=1/4/8|2q<=h|integer scalars|
|e,x,c|専門家番号・特徴・router入力|1|e=0/1, c=(観測手がかり,x)|x in R^d, c in R^(d+1)|index/vectors|
|W,b,w,a,theta|共有重み・バイアス・共有パラメータ|1|theta=(W,b,w,a)|W:hxd,b:h,w:h,a:real|matrix/vectors/scalar|
|m_e,Q_e|専門家コード・回転行列|rad/1|q角度、ブロック回転|m_e in R^q,Q_e:hxh|vector/matrix|
|R(alpha),alpha,I|平面回転・角度・恒等|1,rad,1|R as below|alpha real; matching square I|matrix/scalar/matrix|
|z,phi,Phi|前活性・GELU・標準正規CDF|1|z=Wx+b,phi(t)=t Phi(t)|z in R^h, real scalar t|vector/functions|
|u,beta,p|router重み・バイアス・選択確率|1|p=sigmoid(u^T c+beta)|u in R^(d+1),beta real,p in (0,1)|vector/scalars|
|f_e,y,t,ell|専門家予測・混合予測・教師・損失|1|defined below|real,ell>=0|scalars|
|J_e|共有パラメータに対する予測勾配|1|gradient_theta f_e|same dimension as theta|vector|
|A,D_phi|回転生成子・活性化Jacobian|1|A=[[0,-1],[1,0]],D_phi=diag(phi prime)|matching plane dimensions|matrices|

## Forward and unit check

\[
R(\alpha)=\begin{pmatrix}\cos\alpha&-\sin\alpha\\\sin\alpha&\cos\alpha\end{pmatrix},\qquad
Q_e=\operatorname{diag}(R(m_{e1}),\ldots,R(m_{eq}),I_{h-2q}).
\]
\[
z=Wx+b,\quad f_e=w^T Q_e^T\phi(Q_ez)+a,\quad
p=\frac1{1+e^{-(u^Tc+\beta)}},\quad y=(1-p)f_0+pf_1,\quad\ell=(y-t)^2.
\]

Wx+b and Q_ez have h entries; w^T yields one scalar. The loss subtracts outputs in the same dimensionless units. All mathematical features and weights are dimensionless; angle radians are dimensionless. The code stores W transposed. Experts exclude the router-only mode cue: theta does not affect p.

## Chain rule and starvation boundary

First d ell/d y=2(y-t), d y/d p=f1-f0, and d p/d beta=p(1-p). Multiplication gives
\[
\frac{\partial\ell}{\partial\beta}=2(y-t)(f_1-f_0)p(1-p).
\]
For shared theta, differentiation of the sum gives
\[
\nabla_\theta\ell=2(y-t)[(1-p)J_0+pJ_1].
\]
The private code m1 occurs only in f1, hence
\[
\nabla_{m_1}\ell=2(y-t)p\nabla_{m_1}f_1.
\]
A small p can suppress expert1 even when extra q supplies useful directions. Increasing q does not bound p away from0. Equal p does not equalize J0/J1 norms. The experiment's50/50 warmup is not gradient-norm normalization or PCGrad. Identical expert functions also make the router-bias gradient zero; equal routing and identical functions are different conditions.

## Exact function-preserving extension

cos0=1 and sin0=0, so R(0)=I2. Extend a q1 code to q8 by appending seven0 angles. All new blocks are identity; Q_e is unchanged. Preserve W,b,w,a,u,beta as well. Then z, both f_e, p and finally y are identical for all inputs in exact arithmetic. The measured insertion difference was0. This does not guarantee that future optimizer steps retain low loss.

By this embedding, every q1 function is also in q4 and every q4 function is in q8. Therefore the infimum training loss over the enlarged family cannot increase. Neither convergence of finite-budget Adam nor lower test error follows.

A zero angle is not a dead bilinear-factor path. Differentiating Q^T phi(Qz) at zero on the new plane gives D_phi(z)Az-A phi(z). At z=(1,0), its second component is0.5-Phi(1), nonzero. Residual, gate and readout sensitivity are still necessary for a nonzero loss gradient. A unit test verifies one nondegenerate case, not every possible input.

All widening arms reset Adam, including the narrow shadow. The warm widening arm additionally resets router parameters and is NOT function-preserving as a whole. In the static capacity sweep, extra angles are random; the separate widening test removes that initial-function change.

## Storage, statistics and scope

Counts including router:122/128/136 learned FP32 scalars for q1/4/8;233 for independent experts. A768-byte explicit metadata block produces1256/1280/1312/1700 real bytes. Same-q ordinary gain stores the same count and bytes. Runtime evaluates both experts and padded full-width transforms: no sparse or proportional-q speed claim.

Eight separately seeded replicas have separate tensor slices and Adam moments. Losses are summed across replicas, so each slice receives its isolated mean-batch gradient; a test checks batched versus isolated Adam. Five teacher/data worlds, not40 pseudo-independent datasets, are the replication unit. Absolute variance and variance relative to mean squared are different diagnostics. Small-sample t intervals are descriptive and exclude natural-task extrapolation uncertainty.

This is not a theorem about arbitrary nonlinear Mirrors, deep spline activations, delayed-payoff experts or earlier repository failures. Full derivations, protocols and evidence are retained in the conversation archives.
