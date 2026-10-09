# MA-454 — Context router predicts continuous Mirror code

Status: **FAIL for Mirror-specific advantage**  
Evidence lane: CONTINUOUS_ROUTING / QUALITY / BYTES / COMPUTE  
A1 protocol freeze: `10c9928d`; valid fresh worlds 45420–45422. Initial worlds 45410–45412 are retained as exploratory and excluded.

## H — Hypothesis

A support-context router predicting a continuous two-value Mirror code will recover held-out task functions better than discrete K=8 prototype routing without exceeding its payload or compute budget.

## T — Test

Synthetic scalar regression `y=tanh(a*x+b)`, 32 support values and 256 query values, 64 tasks per world-seed, three valid fresh worlds and three seeds. Context was computed from support-only moments `(mean(x), mean(y), mean(x*y), mean(y²))`. Development compared raw and standardized contexts and ridge λ in `{0.001, 0.01, 0.1, 1.0}`, selecting standardized/0.001. Compared shared-only, discrete K=8 routing, continuous Mirror code routing, an algebraically identical generic linear hypernetwork, and an oracle independent-coefficient upper control. Actual serialized inference packages were measured at N=1/20/64.

The first attempted fresh run skipped the registered development sweep. It was excluded through amendment A1 before replacement fresh worlds were opened. Its result is not pooled with valid results.

## D — FAIL for Mirror-specific claim

Across valid fresh runs, N=20 mean NRMSE was 0.27875 shared, 0.09831 discrete routing, and 0.09553 both Mirror and hypernetwork. At N=20 Mirror/hypernetwork payload was 126.45B/task, versus 97.85B for discrete and 91.65B for the oracle independent upper control. At N=64 Mirror/hypernetwork was 0.08524 / 50.52B per task; discrete was 0.09659 / 31.58B. Mirror and generic hypernetwork had exactly identical outputs and serialized payload hashes in every world/seed/size. Their inference proxy was 520 MAC/task vs 256 for discrete; measured query wall is recorded per run.

Continuous code prediction slightly improves quality over the discrete prototypes, but loses on bytes and is functionally identical to generic linear hypernetwork conditioning. No Mirror-specific benefit was established.

## C — Strongest counter-hypothesis

The Mirror router is simply a two-output linear hypernetwork over support summaries; equal functions, hashes, and bytes directly support this explanation.

## U — Unknown

Richer context encoders, nonlinear hypernetworks, natural tasks, large shared neural modules, and end-to-end support-encoder costs remain untested. The independent control uses teacher coefficients and is only an oracle upper bound, not a fair adaptation baseline.
