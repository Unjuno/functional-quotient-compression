# MA-015 — domain-factorized Mirror experts

Status: SCREENING; development passed and fresh settings frozen.  
Branch: `research/ma-015-domain-factorized-experts-20261007`.  
Base commit: `0f875318639af56f7ff6089cddceaa93d305d480`.

## H — Hypothesis

One shared four-expert bank plus a small view per domain can represent twelve domain×expert functions at lower actual payload than twelve independent MLPs.

## T — Test

Three balanced domains, four oracle top-1 experts per domain, 16→32→16 GELU. The aligned teacher shares an expert bank and applies domain-specific Givens views; an independent teacher measures the private-function boundary. Controls are independent experts, tying, scalar gates, and rank-1/rank-2 residuals. Development world 150000 selects LR; fresh worlds stay sealed until both access conditions pass.

## Development result

- Selected LR 0.003 on world 150000.
- Mirror MSE was 6.34e-4 vs 9.22e-4 independent (0.688x); actual payload was 18,727B vs 51,060B (0.367x). Pre-fresh gate passed.
- Rank-2 residual had MSE 6.13e-4 at 23,270B; hard tying used 18,221B at MSE 1.07e-3. Mirror is close in quality to rank-2 but 19% more bytes than hard tie.
- On unrelated domain/expert functions, Mirror MSE was 2.39e-2 vs 9.57e-4 independent; private state required.

## D — Decision

Fresh results pending; fixed-update development is not capacity evidence.

## C — Strongest counter-hypothesis

The aligned teacher is constructed from the same domain-view structure. Real domains may require independent experts or stronger domain-adaptation controls.

## U — Unconfirmed

Learned routers, natural domains, near-convergence capacity, optimized kernels, and language quality remain untested.

## Fact / interpretation / hypothesis

- **Fact:** one view is shared across expert IDs within each domain; the development access gates passed.
- **Interpretation:** this isolates a domain coordinate crossed with the existing expert coordinate.
- **Hypothesis:** the small domain code may recover useful domain-specific behavior without storing a new expert pool per domain.
