# MA-546 — Functional invariance of hidden activation View symmetries

Status: PASS for the symmetry audit; FAIL as additional functional multiplicity.
Branch: `research/ma-546-representation-symmetry-20261009`
Base commit: `ac1c3e1814aada23eb99254b841faa31b4bb0ffb`
Prior art: PA47 (monomial weight symmetries), PA97 (representation engineering)

## H — Hypothesis

An invertible reparameterization of the post-GELU layer-3 MLP activation, paired with its exact inverse in the down-projection, preserves Pythia's function and adds no logical functions. Omitting the inverse changes the function.

## T — Execution

Pinned Pythia-70M-deduped revision `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`; layer-3 MLP post-GELU width 2048; five worlds (dev 54601/54602, fresh 54611/54612/54613), 128 prompts per world. Tested 16 independent permutation/sign/log-uniform-scale monomial addresses and one dense signed/permuted normalized Hadamard address. Applied each transform after GELU and its exact inverse in the down-projection. Compared canonical model, compensated views, and uncompensated monomial. No parameter training.

The first dev run found an inverse-orientation bug and a raw max-logit gate too strict for float32 tail logits. Amendment 1 fixed the inverse and frozen distribution-level criteria; the first output is preserved under `results/pre_amendment_1/`. Fresh access happened only after the corrected dev run, replay and freeze were committed.

## D — PASS for invariance; FAIL for added multiplicity

Both corrected development seeds and all three fresh seeds pass the frozen compensated-view gates. The transformation codes are gauge labels for the same model function; they do not create additional logical functions.

## Facts

- Across 5 worlds, local down-projection max absolute deltas are <=2.99e-6 for all 16 monomial views and the dense Hadamard view (gate: 1e-5).
- Representative full-model compensated views retain 100% top-1 agreement. Maximum KL across all worlds is 7.91e-6 (gate: 1e-4). Raw maximum logit deltas are 0.0077–0.0122 because floating-point summation order changes.
- Without compensation, mean KL is 0.68–2.18 and top-1 agreement falls to 0.023–0.805 across worlds.
- Actual NPZ state is 231,772 B for 16 monomial views and 15,268 B for one dense Hadamard address, beside 168,144,624 B of shared model files. Sixteen independent model payloads would be 2,690,313,984 B; however they would be duplicate functions in this construction.
- Each world uses 128 prompts. Baseline forward takes 0.42–0.49 s; one compensated monomial full forward 0.44–0.47 s; dense compensation/full forward 0.49–0.67 s; uncompensated monomial 0.44–0.50 s. The 16-view local monomial audit adds 0.00–0.01 s; dense matrix compensation is included in reported dense timing.
- Corrected development seed 54601 replay matched splits, outputs, code arrays, byte counts and metrics exactly (timings excluded). Four tests pass.

## C — Strongest counter-hypothesis

The paired inverse makes the experiment an explicit change of coordinates around one linear map. Equality is therefore a gauge property, not a Mirror-specific capacity gain. The uncompensated transform demonstrates changed outputs but does not show that the change is useful or task-directed.

## U — Not established

The result covers one MLP interface in one Pythia checkpoint. It does not establish which learned activation views preserve functions through residual LayerNorm, attention/rotary paths, cross-layer edits, or task-conditioned interventions. It makes no claim that all activation Views are symmetries.

## Fact / Interpretation / Hypothesis

Fact: compensated activation basis changes preserve the tested function within numerical tolerance; unpaired changes do not. Interpretation: these addresses cannot count as added logical experts or capacity. Hypothesis: other insertion points may have smaller nontrivial stabilizer groups and should be audited separately.
