# MA-266 — composed VeRA codes and Mirror views

Status: SCREENING
Evidence lane: MECHANISM / STORAGE / QUALITY / RUNTIME
Base commit: `af1d096cae5057098f60f91d74be9d382cb598ad`
Prior art: PA18 VeRA; PA26 Task Arithmetic

## H — falsifiable hypothesis

On held-out combinations of two task factors, composing two per-factor orthogonal Mirror views will recover a Givens-composed linear adapter bank with better serialized byte/quality tradeoff than (1) adding factor-specific VeRA-style scaling vectors and (2) the direct non-Mirror sine/cosine coefficient-product control. Unrelated task maps should require private state.

> **Mirror insertion:** this experiment adds two per-factor angle coordinates `m=(α_i, β_j)` to a shared frozen matrix so two logical adaptation factors can compose without storing one adapted matrix per factor pair.

The teacher is a synthetic aligned mechanism, not a trained VeRA model. This is a post-fit/function-family screen, not a claim of learning efficiency or capacity.

## T — frozen screen

Each world has a 12×12 shared core and a 4×4 task-factor cross-product. The aligned teacher is `W_ij = R(α_i) W_core R(β_j)` with fixed, public angle schedule. Even-parity factor pairs are support; odd-parity pairs are held out. Every factor value appears in support. Each pair is evaluated on 128 Gaussian inputs. The codecs see the eight exact full support matrices; evaluation inputs do not train or fit them. Mirror uses the public teacher angle tables as its factor codes (an oracle-aligned upper screen), while VeRA scaling vectors are least-squares fit to support matrices. An unrelated condition replaces the maps with independent matrices.

Methods: shared hard-tied core; full independent 16-map bank (oracle upper); native VeRA-style shared random rank-6 outer-product basis with additive per-factor scaling vectors fit by least squares on support; Mirror two-angle factorized views; and a direct non-Mirror coefficient-product parameterization that stores per-factor `(cos,sin)` pairs and expands the same 2D rotation product. The last control is an explicit functional-equivalence/representation audit; if it ties within the byte gate, no Mirror-specific win is claimed. Factor IDs are supplied to all methods. No learned router.

Development seeds: 26601–26603. Fresh seeds locked but unopened: 26611–26613. No hyperparameter selection is planned; method shapes, rank, split, angle precision and gates are fixed here. Fresh is run only if development passes the full Mirror-specific gate.

## Gates

**Promising / open fresh:** all development worlds have held-out Mirror normalized MSE ≤1e-5 and actual payload ≤50% independent; additionally, Mirror must save ≥10% bytes versus the best non-Mirror method meeting that quality, or achieve ≥10× lower error at no more than 10% extra bytes.
**Fail:** the Mirror quality/byte gate fails, or the non-Mirror VeRA/additive or coefficient-product controls meet the same function/quality at within 10% bytes.
**Not established:** payload replay, serialization, or support/held-out split integrity fails.

## Storage and compute

Actual reloaded deterministic NPZ bytes are authoritative. Charge core, all factor codes, VeRA shared random basis and scaling vectors, coefficient tables, target matrices in the independent upper, metadata and archive headers. Report support examples, zero optimizer updates, arithmetic proxy, encoder time and batched inference throughput. Runtime workspace is separate. CPU only; vendor nanoGPT unchanged.

## C / U

**Strongest counter-hypothesis:** the aligned teacher and Mirror use the same Givens composition chart. The direct coefficient-product model is mathematically the same function written in ordinary coordinates. It may remove Mirror-specificity even if both beat additive VeRA codes.

**Unconfirmed:** trained VeRA composition, input-dependent composition selection, learned task/router inference, natural language, nonlinear networks, quality at near-converged fixed-byte frontiers and optimized deployment kernels.

## Decision

FACT: see result table and verification.
INTERPRETATION: separate compositional generalization from Mirror-specific benefit.
HYPOTHESIS: factorized angles may encode useful held-out compositions compactly.
BOUNDARY: no general adapter-composition, capacity or language claim.

## Results — development-only decision

### H / T / D / C / U

**H:** two factor-specific orthogonal views would generalize from checkerboard support pairs to held-out compositions more efficiently than VeRA-style additive scaling coordinates; unrelated functions would need private state.

**T:** three development worlds (26601–26603), 4×4 task-factor cross-product, eight even-parity support maps and eight odd-parity held-out maps, 128 evaluation inputs/map, 12×12 matrices, exact support matrices supplied and zero optimizer updates. Controls: tied core; rank-6 shared random VeRA-style outer-product basis with additive factor codes; direct sine/cosine coefficient-product expansion; independent full maps. Six conditions × methods/splits produced 60 rows. Factor IDs were supplied. Fresh seeds 26611–26613 stayed sealed because the preregistered Mirror-specific development gate failed.

**D: FAIL for Mirror-specific value under the frozen gate.**

**Facts:** On aligned held-out pairs, Mirror had mean normalized MSE **7.91e-9** at **1,276B**; the direct coefficient-product control had **4.20e-8** at **1,304B**. The coefficient control is only 28B (2.1%) larger, below the required 10% byte margin; Mirror's error advantage was 5.3×, below the required 10× quality margin. Both were under the 1e-5 functional-quality threshold. The coefficient-product table stores per-factor sin/cos pairs and expands the same rotation product. Native additive VeRA-style scaling was materially worse on held-out aligned pairs (normalized MSE **0.289**, 2,250B); hard tying had **0.293**, 818B; independent full maps were near zero, 9,464B.

On unrelated held-out maps, Mirror and coefficient-product controls both had normalized MSE about **0.401**; VeRA-additive and hard-tied controls were about **0.154** and **0.150**; independent maps were near zero. Thus the compact composition chart did not recover unrelated behaviors.

The frozen active-operation proxy was 432/example for Mirror versus 732 for coefficient-product reconstruction, and development batched CPU throughput averaged 1.77M versus 1.15M examples/s on aligned held-out pairs. This is an implementation-sensitive timing signal from a small NumPy harness. It does not erase the exact functional alias: a more efficient coefficient-product implementation can apply the same two rotations. No runtime claim is established.

Four tests passed. All 60 result rows, serialized sizes, payload hashes and deterministic metrics replayed exactly (max difference 0). The failed implementation attempt before the final screen used an incomplete coefficient expansion that rotated only two channels; it was caught by the equivalence test, was not analyzed, and was replaced before the recorded development run. The valid control applies the same rotation to all six 2D channel blocks as Mirror.

**Interpretation:** The experiment supports held-out composition for a known Givens family and shows why simple addition of VeRA-like scaling codes cannot express the interaction. It does not show that Mirror is needed: ordinary sine/cosine coefficients express the same function at nearly identical payload size. The measured operation proxy favors the optimized geometric form over this particular expanded coefficient implementation, but the control can use the same transform algorithm.

**C — strongest counter-hypothesis:** The target function is generated by the same rotation group used by the Mirror model. Its coefficient-product expression is an exact algebraic rewrite with nearly identical bytes. The small code-size difference comes from storing one angle instead of a sin/cos pair and may not survive quantized coefficient storage, packing, or metadata changes. The arbitrary-map stress condition confirms the benefit depends on alignment.

**U:** Fresh replication, learned factor codes, trained VeRA, input-conditioned routing, other code families, nonlinear adapters, natural-language behavior, and optimized deployed kernels remain untested. Because the direct coefficient control satisfied the same quality within 10% bytes, no fresh run was opened under this protocol.

FACT: see `RESULTS_CORE.csv` and `source/replay_verification.json`.
INTERPRETATION: composition generalizes for the aligned chart; Mirror-specific margin fails.
HYPOTHESIS: nonlinear group-coordinate composition may save more state for larger factor tables, but that is a new protocol.
BOUNDARY: synthetic post-fit functions; no capacity claim.
