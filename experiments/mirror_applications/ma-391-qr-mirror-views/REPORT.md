# MA-391 report

## H — Hypothesis
A factorized address m(q,r)=alpha_q+beta_r over two partition tables can recover useful held-out quotient/remainder embeddings beyond fixed addition, multiplication, or concatenation, at lower cost than independent token rows. This is falsifiable by the preregistered two-world quality, payload, uniqueness, and runtime gates in `PROTOCOL.json`.

## T — Trial
A synthetic 12×12 token space, 16D embedding, and 8-class label task. Each teacher used partition vectors and factorized Givens rotations. Exactly 36 of 144 q×r pairs per world were excluded from training; each individual q and r remained represented. Six methods were trained for 512 Adam updates on 65,536 examples per method, using seeds 39101 and 39102. Controls were independent table, addition, multiplication, concatenation, and free per-pair angle diagnostic. Actual deterministic ZIP/NPY FP16 payloads were loaded before validation/test scoring. CPU throughput used 100 repeated batches of 9,216 lookups per method and seed.

## D — Decision
**FAIL.** Fresh seeds 39111–39113 remain sealed. Mirror uniquely represented all 144 combinations, but missed multiple gates.

| Seed | Mirror seen acc. | Mirror heldout acc. | Concat heldout acc. | Mirror bytes / concat | Mirror CPU / concat |
|---:|---:|---:|---:|---:|---:|
| 39101 | 0.9623 | 0.5557 | 0.5000 | 2292 / 1802 | 0.332× |
| 39102 | 0.9810 | 0.4116 | 0.5251 | 2292 / 1802 | 0.282× |

The best native seen-pair accuracy also exceeded Mirror by 3.77 points in seed 39101 (multiplication: 1.0000 vs 0.9623), beyond the allowed 2 points. Mirror payload was 27.2% larger than concatenation, despite remaining well below the independent table (5,804B).

## C — Strongest counter-hypothesis
The one-world held-out gain reflects a teacher-aligned synthetic construction plus sampling variability. The second world reverses the result; the Mirror implementation spends more bytes than the strongest fixed baseline and computes much more slowly because of indexed angle lookup and trigonometric composition.

## U — Unconfirmed
No natural language, large vocabulary, near-convergence, accelerator kernel, or fresh-world result is established. The free per-pair angle is diagnostic and is not evidence for held-out transfer.

## Evidence classes
**Facts:** all listed measurements, serialized bytes, hashes, and verification results are recorded in `runs/`, `RESULTS_CORE.csv`, and `VERIFICATION.json`.

**Interpretation:** the preregistered development screen fails quality consistency, storage relative to best fixed composition, and runtime.

**Hypothesis:** a structured view can help held-out combinations only where the task reliably shares that view algebra, and its inference implementation must be fused to avoid the trigonometric runtime penalty.
