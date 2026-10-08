# MA-818 — Mirror code for adding a new operation

Status: FAIL for Mirror-specific advantage; narrow mechanism PASS
Evidence lane: MECHANISM / STORAGE / RUNTIME
Base commit: 28194ea (research/mirror-application-worker-ready-20261007)

## H — hypothesis

A new operation on already represented content can be added by a compact operation-only Mirror code without changing the content representation; its held-out behavior is compact relative to a full operator, while an ordinary operation embedding may match it exactly. A private full operator may be needed for unrelated dynamics.

## Mirror insertion

> **Mirror insertion:** this experiment adds an operation code m to the shared content-to-state transform so that a new logical transformation can act on existing content vectors without storing a full new operator matrix.

The old operation bank and content vectors stay fixed. The new operation code is inferred from support examples. All codes, old operation entries, residuals and private matrices are charged.

## Prior-art delta

PA217 separates computational content (what) from the mechanism (how) so each can be extended independently. MA-818 isolates operation growth while holding content fixed; MA-817 is a separate content-growth hypothesis and is not run here. Controls are an ordinary linear operation embedding/low-rank adapter and an independent full matrix. The new operation is tested both on the existing Givens family and as an unrelated dense transform to expose the private-dynamics boundary.

## Task and controls

A frozen 8-D content representation is transformed by four independent 2-D Givens operations. Three old operations are already stored. The new aligned operation is inferred from 4, 8 or 16 support content/output pairs; held-out content vectors evaluate it and compositions. An unrelated dense operation is a private-state diagnostic. Methods: structured Mirror angle code, exact ordinary structured operation embedding (same code/function, required null control), ordinary linearized generator-code adapter, and a full fitted operator.

No optimizer updates. This is a synthetic representation/function interface, not a recurrent neural network or learned what/how system.

## Gates

PASS for the aligned mechanism requires 3/3 fresh worlds with held-out normalized MSE <=1e-5 at support 8, payload <=0.90x the full matrix adapter, and old-operation retention error <=1e-5.

Mirror-specific pass additionally requires >=10% fewer bytes than the exact ordinary structured operation embedding at matched quality. That control uses the same four operation coordinates and is expected to expose whether this is simply a standard operation embedding.

The unrelated dense diagnostic passes only with private full operator state. This boundary diagnostic is distinct from the Mirror-specific claim.

## Random selection

Draw 11 selected MA-818 from 452 P0/UNTESTED candidates at index 316. Pool and hashes are recorded in PROTOCOL.json and source/selection_pool.csv. No MA-818 branch was found.

## H / T / D / C / U

H: A new aligned operation can be fitted from support pairs with a compact operation-only code and without changing the frozen content or old-operation bank. For an unrelated operation family, compact Givens state should fail and the full private operator should recover it.
T: three fresh synthetic worlds (81811–81813), 4/8/16 support pairs, 64 held-out content vectors per world, four-step compositions, exact ordinary structured embedding, linearized generator code, and independent full operator. No learned optimizer or model updates. Actual serialized payload bytes and CPU wall time were recorded.
D: **FAIL for Mirror-specific advantage; PASS for the narrow aligned mechanism and private-state boundary.** At support=8, Mirror reached normalized trajectory MSE 1.62e-15 in all 3 worlds at 290 B versus 531 B for the full operator, with old-operation retention error below 1e-15. The ordinary structured operation embedding had identical bytes, payload hashes, and quality in all 18 paired fresh conditions. For unrelated dynamics at support=16, compact Givens MSE averaged 1.03; a 534 B private matrix reached 1.20e-15. See RESULTS_CORE.csv and source/fresh.csv.
C: The result is standard low-description operation embedding of an aligned transform family; calling that code Mirror adds no measurable effect. The measured CPU timing is tiny synthetic NumPy work and does not establish deployment latency.
U: learned content representations, recurrent neural dynamics, noisy few-shot operation inference, continual retention over sequential operation additions, and language tasks.

## Fact / Interpretation / Hypothesis

FACT: Fresh aligned quality gate passed 3/3 and payload used 45.4% fewer bytes than the full fitted operator at support=8. Mirror and ordinary structured embedding had exactly identical payload hashes and metrics. Unrelated Givens views missed badly; private full operators fit the unrelated task at support>=8.
INTERPRETATION: A fixed structured operation family supports compact operation growth on frozen content. This evidence does not isolate a Mirror-specific advantage; an ordinary structured embedding is the same representation and function at the same actual bytes. Unrelated operation growth crosses a private-state boundary.
HYPOTHESIS: Neural what/how systems may benefit from compact operation addresses only when new procedures lie in a known structured family; broad operation growth likely needs private residual or operator state.
BOUNDARY: synthetic linear content transformations only.
