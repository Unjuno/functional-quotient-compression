# MA-878 status

- Status: FAIL
- Dedicated branch: `research/ma-878-head-matched-cachebridge-20261111`
- Protocol frozen: `e6a164427071f3b424d823e727e4a1f96eb6e9c3`
- Fresh rank/splits frozen: `e9b36b275`
- Fresh worlds: seeds 983, 997, 1009; validation calibration / test evaluation
- Fresh result recorded; actual serialized payload measured
- Verification: prompt IDs and payload accounting PASS; preregistered quality gate FAIL
- Tests: 3 passed; syntax compilation passed
- CUDA unavailable; CPU-only scope

## H — falsifiable hypothesis

Attention-weighted injective source-head selection improves fresh NLL over fixed spread by at least 0.10 nat in all three seeds; rank-32 shared map coding stays within 0.05 nat of full maps and 0.50 nat of native while using at most half the bytes.

## T — protocol and execution

See `PROTOCOL.json`, `FROZEN_SETTINGS.json`, `dev_results.json`, and `fresh_results.json`. The development rule selected rank 32 before opening fresh results. Three fresh seeds used 12 calibration and 16 test examples each. Pythia-70M→SmolLM2-135M, Wikitext-2, position-free inverse-RoPE map interface. Full details and timings are in `RESULTS.md`.

## D — decision

FAIL. Actual rank-32 payload was 573,125 B versus 2,996,165 B full-map payload (80.871% reduction), but rank 32 was +14.522/+10.446/+7.478 nat over native. Attention matching beat fixed spread by +0.158 nat in one seed, then lost by 0.384 and 0.217 nat in the other two. All preregistered quality gates fail.

## C — strongest counter-hypothesis

The architectures have incompatible cache representations; support selection helps inconsistently, and rank-32 PCA discards target-relevant directions. Full affine maps also remain far below native, indicating that head assignment alone is not the main limitation.

## U — remaining uncertainty

One ordered model pair and corpus; no GPU, generation rollout, nonlinear bridge, larger calibration budget, or paper-faithful CacheBridge reproduction. Results do not establish a general limit for head matching.

## Fact / interpretation / hypothesis

- **Fact:** measured NLL, serialized bytes, prompt identities and CPU times are in the result artifacts.
- **Interpretation:** the bridge has a large byte-quality tradeoff here; attention-matched support has no reliable advantage over fixed spread.
- **Hypothesis:** a materially richer/private mapping or closer model pair may be needed for useful transferred KV state.
