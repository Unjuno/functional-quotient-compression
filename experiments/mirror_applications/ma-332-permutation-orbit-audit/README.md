# MA-332 — Mirror permutation-orbit audit

Status: FAIL as a source of additional functional capacity; symmetry audit passed. Dedicated branch: `research/ma-332-permutation-orbit-audit-20261008`.

## H — hypothesis

A consistent hidden-unit permutation changes a trained MLP's parameter coordinates while leaving its input-output function invariant. Charging permutation indices may compress duplicate checkpoint storage, but it does not create new functions.

## T — execution

Trained one small 8-12-3 ReLU MLP for 300 Adam updates on a deterministic synthetic regression task. For each development/fresh seed, generated eight hidden-unit permutations. Permuted incoming weights, hidden biases and outgoing rows consistently. Compared eight separately serialized permuted checkpoints with one base checkpoint plus eight paid permutation index vectors. A negative control permuted incoming tensors only.

Seeds: development 33201/33202; fresh 33211/33212/33213. Inference outputs were measured after FP16 serialization reload. Total shared payload includes every index vector and archive metadata.

## D — decision

**FAIL as additional logical function capacity.** All eight permutation views are the same function to numerical tolerance. Across five seeds, max output difference stayed in 1.1e-7–2.3e-7. The shared base plus paid indices used 1,740B versus 9,440B for eight separate checkpoint archives (81.6% fewer bytes), but it still represents one function eight ways. The incoming-only negative control changed outputs substantially (max difference 10.0–18.8), confirming that the full symmetry requires permuting all coupled tensors.

The audit objective itself passed: permutation-only changes must be treated as zero functional multiplicity.

## C — strongest counter-hypothesis

A low-bit or delta-coded archive of eight identical checkpoints could also reduce storage. This experiment only compares deterministic NPY archives, so the 81.6% duplicate-storage reduction is not a Mirror-specific codec advantage. More importantly, none of the saved bytes add a distinct function.

## U — unresolved

This covers hidden-unit permutations in one trained synthetic ReLU MLP. It does not test sign/scale symmetries, other architectures, natural checkpoints, or a codec optimized for duplicated tensors.

## Fact / interpretation / hypothesis

**Fact:** Five independent seeds and eight permutations per seed were run. Shared checkpoint-plus-index bytes were exactly 1,740B; eight separate archives totaled 9,440B. Max output difference was <=2.25e-7. Ten serialized rows replayed byte/hash and metric values exactly; four tests pass.

**Interpretation:** Parameter-space multiplicity under a known symmetry is not functional multiplicity. The permutation address needs no independent logical-capacity credit.

**Hypothesis:** Every proposed View that overlaps a known symmetry group needs this exact function-orbit audit before any expert/capacity count is assigned.

## Evidence files

- `RESULTS_CORE.csv`: development and fresh rows.
- `artifacts/`: serialized checkpoints and metrics.
- `source/verify.py`: payload and metric replay.
- `VERIFICATION.json`: verification record.
