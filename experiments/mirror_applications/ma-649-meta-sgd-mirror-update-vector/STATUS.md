# MA-649 status

- Status: FAIL
- Branch: `research/ma-649-meta-sgd-mirror-update-vector-20261104`
- Base commit: `35be5784`
- Protocol frozen before development: yes
- Development complete: yes (corrected estimator, seeds 64901/64902)
- Fresh/audit opened: no; frozen failure gates reached

## H / T / D / C / U

**H:** A shared rank-4 basis with per-family Mirror codes compresses Meta-SGD update vectors for correlated task families while retaining one-step query quality; increased heterogeneity reveals where private update coordinates are necessary.

**T:** 32D diagonal quadratic tasks, 8 families, 256 support and 512 query examples/family, heterogeneity 0/.25/.5, two development seeds. Compared global Meta-SGD, independent per-family full vectors, rank-4 Mirror/PCA, and rank-4 random-subspace codes. Used analytic curvature-weighted least-squares fit (0 gradient optimizer updates); serialized complete inference payloads with initialization and family routing state; measured actual bytes and query loss. Fresh seeds 64911–64913 remain sealed.

**D:** FAIL. At heterogeneity 0, Mirror quality is close to independent Meta-SGD (loss ratio .999) but actual payload is 1,834–1,835B vs 1,910–1,913B, only ~4% smaller; this misses the <=75% gate. At heterogeneity .25, Mirror loss is 1.56–1.72x independent; at .5, 6.37–9.08x. Mirror rank-4 basis/codes exactly equal native PCA functions; byte deltas are at most a few bytes from container serialization. Random subspace is substantially worse. Fresh stayed sealed.

**C:** Family-specific update vectors mostly occupy a low-dimensional correlated subspace, but NPZ/metadata overhead dominates the small 32D bank. At higher heterogeneity, rank4 truncation loses task-specific update directions. Ordinary PCA explains the Mirror reconstruction exactly.

**U:** This is an analytic toy few-shot adaptation test, not a trained Meta-SGD or language model. Broader task distributions, initialization learning and multi-step adaptation remain untested. The first implementation omitted curvature weighting in the meta-fit denominator; corrected results replace it, and the buggy attempt is preserved separately.

## Fact / Interpretation / Hypothesis

- **Fact:** Corrected run has 30 rows; three tests pass; serialized byte and query metric replay pass; protocol hash matches freeze. Fresh seeds untouched.
- **Interpretation:** Low-rank coding tracks correlated update vectors but fails the required actual-byte frontier and collapses exactly to native PCA; private directions matter as task heterogeneity rises.
- **Hypothesis:** Higher-dimensional or larger family banks may amortize the basis overhead, but quality must beat matched native PCA at equal bytes and realistic few-shot task behavior.
