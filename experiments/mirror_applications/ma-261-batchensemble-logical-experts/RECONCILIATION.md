# MA-261 result reconciliation

## Fact

Two dedicated source branches contain separate MA-261 protocols:

- `research/ma-261-batchensemble-logical-experts-20261008` records a post-fit, oracle-routed 8D-to-6D screen with four fresh seeds. Its source README reports that the aligned gate passed 4/4. However, the frozen protocol requires Mirror MSE <=1.10x the independent model per fresh world. Recomputing from its stored `RESULTS_CORE.csv` gives ratios 1,390x, 302,057x, 11.7x, and 0.227x. Thus the literal gate passed only 1/4. Payloads were 1,006B Mirror, 1,562B BatchEnsemble, and 1,794B independent; absolute aligned Mirror MSE was tiny, but that does not satisfy the stated relative gate.
- `research/ma-261-rankone-mirror-experts-20261008` records a different 1,200-update two-expert regression screen. It failed its development quality gate; fresh worlds 26102–26104 remained sealed. Its metric replay was exact and two tests passed.

The two tasks, parameterizations, and optimization protocols are not replications. Both reports are retained separately under `protocol_variants/fixed_update_failure/` and the root post-fit report.

## Disposition

MA-261 is **FAIL** against the registered frozen gate. The post-fit result shows descriptively tiny absolute error and lower serialized state on a deliberately Givens-aligned teacher, but it failed the per-seed relative-quality requirement in 3/4 fresh worlds. The fixed-update protocol also failed during development. The original branch's claim “PASS 4/4” is preserved on that branch; this integrated record corrects it from the protocol and raw metrics rather than silently changing the protocol.

## H / T / D / C / U

- **H:** one shared expert plus per-role Givens coordinates can replace BatchEnsemble rank-one factors for aligned routed roles at lower bytes while retaining the preregistered quality bound.
- **T:** post-fit least-squares screen, oracle sign router, development seeds 17/31, fresh seeds 107/227/311/419; zero optimizer updates. A distinct 1,200-update screen used worlds 26100/26101 and remained fresh-sealed after failure.
- **D:** FAIL; strict relative-MSE gate passed only 1/4, plus a separate development FAIL.
- **C:** the independent control is at numerical zero, making a relative-error gate unstable; nevertheless the gate was frozen and must be applied as written. The positive branch's pass summary appears to use a different, undisclosed tolerance.
- **U:** natural expert variation, learned routing, deeper nonlinear experts, robust absolute/relative quality criterion, and runtime at scale.
