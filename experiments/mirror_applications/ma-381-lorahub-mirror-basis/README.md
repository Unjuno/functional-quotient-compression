# MA-381 — LoRAHub over Mirror-compressed candidate basis

Status: PROMISING only for the aligned synthetic candidate bank; registered strict gates missed, fresh remains sealed.
Evidence lane: QUALITY / STORAGE / COMPUTE / LOW-RANK COMPOSITION
Base commit: `bc86da2`
Doctrine: `docs/phase2/MIRROR_PARAMETER_INTEGRATION_DOCTRINE.md`
Integration map: `docs/phase2/MIRROR_PARAMETER_INTEGRATION_MATRIX.md`

## H — Hypothesis

For candidate LoRA updates on a common Givens orbit, one rank-2 physical basis plus one angle per candidate will preserve few-shot signed LoRAHub composition quality with fewer serialized bytes than the full bank and beat a scalar shared-basis control. Unrelated candidates provide the registered boundary condition.

## Mirror insertion

> **Mirror insertion:** this experiment adds a source-task angle to conjugate a shared rank-2 LoRA update, giving eight candidate modules to the LoRAHub coefficient search without storing eight separate A/B factor pairs.

- Native method: LoRAHub signed composition over eight candidate LoRA modules.
- Exact insertion: Givens conjugation of the reconstructed rank-2 update for each candidate task.
- Persistent `m`: eight source angles, plus few-shot target composition coefficients.
- Sweep: fixed rank-2 shared basis, eight candidates, 32 support examples, no hyperparameter selection.
- Cheapest ordinary control: eight direct scalar source coefficients.

## Physical-to-logical claim

- Physical object: one pair of shared rank-2 LoRA factors.
- View coordinate: one angle per candidate module.
- Logical multiplicity: eight source modules composed into four target updates.
- Storage premise: replace eight A/B pairs with one basis and eight small codes.
- Failure mode: unrelated candidate subspaces may not lie on one rotation orbit; ordinary scalar codes may match any gain.

## Prior-art delta

Read the registry row and referenced PA items first.

- Closest prior art: PA55 LoraHub.
- Prior art establishes: positive/negative scalar composition of candidate LoRA modules for few-shot tasks.
- Exact Mirror delta: reduce candidate-bank bytes before the same signed composition search.
- Cheapest ordinary control: scalar shared-basis module codes.

## T — comparisons and protocol

Compare independent LoRAHub candidates, hard-shared one candidate, scalar shared basis, and Givens Mirror shared basis. Evaluate both an aligned Givens-orbit candidate family and unrelated rank-2 candidate updates. Fit shared candidates from source-task examples; estimate all target composition coefficients from the same 32 support examples using ridge least squares.

## Gates

### PASS
Both aligned development seeds meet the full quality/byte gate in `PROTOCOL.json`; then fresh aligned and unrelated worlds may open.

### FAIL
Mirror misses aligned target quality or byte limits in both development seeds, or scalar matches the registered quality margin.

### NOT ESTABLISHED
Full candidate LoRAHub does not establish low target MSE; fresh remains sealed.

## Tuning boundary

Development: seeds 38101/38102; both candidate conditions registered before runs.

Fresh: seeds 38111/38112/38113; never used for tuning.

## Storage contract

Pay candidate A/B factors or shared basis, source codes, few-shot fusion coefficients, IDs, metadata, archive headers, and one-time View reconstruction compute. Actual ZIP/NPY FP16 payload bytes are authoritative.

## Compute contract

Report 1,600 source updates, source/few-shot examples, ridge solve MACs, candidate encoding MACs, one-time decode MACs, wall clock, and target throughput.

## Results

### D — decision

**Fact:** In aligned worlds, Mirror target test MSE was 4.59e-10 and 2.30e-8, versus scalar 7.24e-4 and 1.96e-3, and full LoRAHub 2.20e-10 and 2.87e-9. Mirror is close in absolute error but misses the strict 1.10x relative-quality gate in both seeds. Its 2,298B payload is 0.775x the full 2,964B bank, missing the registered 0.60x byte target; it equals the scalar payload. Hard sharing is 2,060B but has worse target error (6.76e-4 / 1.91e-3). In unrelated worlds, Mirror target MSE was 0.0199/0.0263 versus independent 1.69e-9/2.90e-9; private candidate modules are required for these arbitrary updates. One-time Mirror decode is 98,304 MACs, in addition to 640 candidate-encode MACs per example. All 16 payloads passed hash/size/reload/metric replay; four tests passed. Fresh remains sealed.

**Interpretation:** This is a PROMISING, development-only aligned feasibility signal: Givens coordinates captured eight deliberately related candidate modules in one rank-2 basis and strongly beat the same-byte scalar basis. The view did not reach the registered storage or relative-quality margins, and its decode slowed inference. Unrelated task updates remained far outside the shared orbit and need private state.

### C — strongest counter-hypothesis

The aligned source family is constructed exactly on the tested Givens orbit, so it demonstrates a mechanism-level feasibility case rather than prevalence. Relative errors against a nearly zero independent baseline magnify tiny absolute deviations; nevertheless the preregistered relative gate is missed and cannot be relaxed after seeing results.

### U — unresolved

Natural LoRAHub candidate tasks, noisy few-shot labels, larger ranks, search-cost scaling, more efficient fused decoders, and fresh confirmation are untested. No general LoRAHub or capacity claim is made.

### Evidence labels

- **Fact:** actual payloads, hashes, per-task MSE, timings, and compute proxies are in `results/development/` and `RESULTS_CORE.csv`.
- **Interpretation:** structured views work on this aligned synthetic orbit, while unrelated candidate functions need private modules.
- **Hypothesis:** clustering real candidate LoRA deltas before assigning shared bases may expose useful compression regions; this requires new preregistered evidence.
