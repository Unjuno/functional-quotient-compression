# MA-1175 — One heavy forward, multiple logical Mirror outputs

**Status: UNTESTED (full scientific hypothesis).** Stage-0 exact/non-exact mechanics reported separately; neither adoption nor trained-LM gains established. Prior art: PA16 Parameter Superposition; PA42 MIMO; PA417 GVA; PA414 Transformer gauge equivalence. Nearest existing MA-310, MA-351, MA-691, MA-241, MA-253, MA-231.

| 記号 | 意味（日本語） | 単位(SI) | 定義・範囲・型 |
|---|---|---|---|
| x | トークン/hidden入力 | 1 | 実行ベクトル、幅d |
| z | 共有preactivation | 1 | z=xU、長さhの実行ベクトル |
| U,D | FFN共有射影 | 1 | U∈R^{d×h}, D∈R^{h×o} 実行列 |
| m | Mirror機能座標 | 1 | 有限次元実ベクトル、回転角はrad |
| T_m,R_m | hidden / output変換 | 1 | h×h または o×o、可逆 / 直交行列 |
| φ | GELU非線形 | 1 | 成分ごとR→R |
| f_m | 第m論理出力 | 1 | R^d→R^o の関数 |
| M | 論理View数 | 1 | 正整数、1/2/4/5/8 |
| S | 直列化容量 | byte (実用単位) | 0以上整数、base/code/decoder込み |
| C, T_io | compute/転送時間 | s | 非負スカラー、デバイス/帯域依存 |
| L | 教師との予測誤差またはNLL | 1 (nat/token) | 有限非負スカラー |
| u_seed,u_eval,u_num,u_c,U_exp | 標準/拡張不確かさ | Lと同単位 | 非負実数、相関考慮 |
| k_cov | 包含係数 | 1 | 正実数、暫定2 |

## H — falsifiable

At **matched downstream task quality** and all inference bytes, one shared heavy execution plus cheap per-View transforms can generate 4 or 5 distinct *useful* logical specialist outputs faster than the best independent folded expert bank, native MIMO/BatchEnsemble/gate or output-linear-head control without increasing P95 runtime by over 10% relative to the fastest byte-feasible competitor. An algebraically exact orbit that simply changes basis is not independent capacity.

## Mirror insertion

> **Mirror insertion:** add m at a declared cached-statistic interface r(x) (output-only, hidden preactivation, or canonical KV readout), obtaining f_m(x)=g_m(r(x)) at lower incremental computation than rerunning the whole specialist.

Output-only case: f_m(x)=f_0(x)R_m. Internal nonlinear case: with row vectors z=xU, f_m(x)=GELU(zT_m)T_m⁻¹ D. Source shared U is evaluated once, but GELU and D remain View-specific. With f_0(x) alone, exact reuse exists **iff** f_0(x)=f_0(x') implies f_m(x)=f_m(x') for every valid pair. Proof follows from function factorization and well-definedness on each fiber. Pure permutation commuting through GELU produces identical function, not diversity.

**Unit check:** x(1×d)U(d×h)T_m(h×h)GELU(·)(1×h)T_m⁻¹(h×h)D(h×o) returns 1×o; each quantity dimensionless; S in byte and timing C in seconds are separate objectives.

## T — worker-independent implementation

1. Freeze shared 2-layer 32/64 width toy Transformer or FFN and 4/5 independently supervised task identities. Two teacher families: true output-orbit and hidden nonlinear View (aligned), plus unrelated independently trained specialists (adversarial).
2. Repeat source shared-forward only vs per-view full recompute vs cached-preactivation exact vs output-only learned readout, ordinary linear/FiLM/IA3, MIMO and independent folded full experts. Fix task trainer, source calibration, per-task support budget and same optimizer updates.
3. Train all Views jointly or accumulate gradients serially and compare ∇shared numerically; do NOT confuse packed execution with reduced arithmetic.
4. On development seeds 11–13 select code breadth r, Mirror count M and decoder capacity; fresh seeds 101–105 test full heldout task identities. Independent deployment replication >=10 new seeds required before adoption.
5. Measure task NLL, worst-task quality, output functional independence (heldout output Jacobian rank), old-task retention, softmax/cache provenance, actual serialized NPZ/safetensors bytes, P50/P95 real CPU/GPU throughput and MAC/FLOP proxy. Stage-0 source/metrics in [report](../../REPORT.md).
6. Before deployment, compute exact output-only collision witness, invalid key-only attention transformation counterexample, RoPE commutant check, and memory alias/copy checks. Never call these extra functional capacity.

## D — decisions

**PASS:** fresh >=4/5 independent task worlds meet no worse than +0.02 nat/token NLL vs strongest non-Mirror same-byte baseline, at least 10% total persisted expert bytes saved and <=0.90 paired P95 of best same-quality folded/control, without hiding view decoder or routing memory. A K4/K5 synthetic wall win alone is not sufficient.

**FAIL:** Any native cheap output head or same-byte factorized bank Pareto dominates, output-only transport distorts nonlinear tasks, or folded independent/tied experts are faster at admissible RAM. K4 Stage-0 CPU gate failed; K5 stage-0 cached-preactivation gain is contingent and folded CPU wins in separate Stage-1.

**UNCERTAIN:** Task IDs/outputs leaked, native trained controls unavailable, GPU transfer and real NLL not measured, or five-world uncertainty too wide.

## C — how hypothesis breaks

Internal Mirror before GELU does not commute with GELU. With lossy final output y0, different original inputs may map to the same y0 but distinct role outputs; then no decoder of y0 alone can be exact. A larger cached z rescues exactness but must still pay all View-specific nonlinear kernels. Pre-folding saves runtime at high resident weight bytes.

## U — uncertainty

u_c=sqrt(u_seed²+u_eval²+u_num²) only for uncorrelated quality estimates; include covariance otherwise. U_exp=k_cov u_c with k_cov=2 is indicative at n=5, not guaranteed 95%; group-bootstrap independent whole tasks. Numerically verify FP64 and deployed dtype. Report separate seconds-based clock repeat error and exact byte sizes.

## Evidence and required next action

Read [Stage-0 report](../../REPORT.md), source archive and two frozen protocols, then implement an independent learned multitask specialist teacher. Do not change worker status based only on source-aligned toy.
