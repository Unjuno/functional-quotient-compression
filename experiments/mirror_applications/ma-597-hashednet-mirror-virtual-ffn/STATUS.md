# MA-597 status

- Status: **FAIL** (development screen; fresh stayed sealed)
- Branch: `research/ma-597-hashednet-mirror-virtual-ffn-20261009`
- Protocol/source frozen before training.
- Development seeds 59701/59702 complete; deterministic replay exact for six methods each.
- Fresh seeds 59711–59713 unopened because Mirror missed the native quality-improvement gate and byte-near simple controls matched or beat it.
- Mirror Givens: 97.56%/98.22%, 9,753 B; native HashedNet: 97.56%/98.67%, 9,441 B.
- The 4,096-bucket native control reached 98.44%/98.67% at 13,537 B and reduced collision rate.

## H / T / D / C / U

- **H:** 32 Givens angles improve collision-limited hashed FFN accuracy at near-equal bytes and beat diagonal/low-rank controls.
- **T:** 64→128→10 MLP on sklearn digits; two stratified dev splits; six methods; 800 updates; all inference payloads round-tripped.
- **D:** **FAIL.** Mirror did not beat native by 1 point on either seed and was matched/beaten by diagonal/rank-one controls at ≤1.10× bytes.
- **C:** Native bucket expansion, extra gates, and rank-one residual explain or exceed the effect.
- **U:** Fresh splits and broader models/tasks remain untested.

See `RESULTS.md` for separated Facts / Interpretation / Hypothesis.
