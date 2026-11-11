# MA-848 — FAIL for Mirror-specific attribution

## H / T / D / C / U

- **H:** A persistent task coordinate composed with transient Hebbian episode state retains unseen task×memory combinations while using less storage than independent pair matrices.
- **T:** Synthetic D=16, 8 task gains × 6 episodes, 24 seen and 24 checkerboard-heldout pairs; one-shot 16-key support per episode. Development seed 17, fresh seeds 37 and 59. Controls: full effective matrix per pair, native task-code × fast-state factorization, Mirror task View × episode View. No optimizer updates; 96 support examples/world.
- **D:** **FAIL** for Mirror-specific value. Both factorized methods reach heldout nRMSE 0 and exact state reset, using 6,760 B for native factorization and 6,764 B for Mirror versus 49,488 B for independent matrices (86.3% fewer serialized bytes). Native and Mirror have identical measured query MACs (104,448 per heldout set); Mirror's extra 4 bytes are NPZ key-name overhead. The factorized methods perform 6.25% more query MACs than the full pair-matrix control because each query also applies its task gain.
- **C:** Ordinary factorization of persistent task identity and transient fast state fully explains compositional recovery and storage savings.
- **U:** Learned task inference, nonlinear plasticity, noisy keys, long episodes, real training/runtime, and whether a learned Mirror code can outperform native factorization.

## Corrected facts (Amendment 01)

Across fresh seeds 37 and 59, heldout nRMSE and reset residual are both 0. Serialized payloads are 49,488 B full pair matrices, 6,760 B native factorized bank, and 6,764 B Mirror bank. One active writable fast state is 1,024 B; the six-state serialized bank is 6,144 B. Hebbian writes cost 24,576 MACs. Query cost over 24 heldout pairs × 16 queries is 98,304 MACs for full matrices and 104,448 for either factorized method. NPZ payloads are reloaded and tensors compared exactly. The same writable fast-state object is zeroed in place between episodes.

## Interpretation

The result establishes a useful synthetic storage frontier for factorizing repeated task×episode effective matrices, but not a Mirror-specific gain: native factorization has identical outputs and compute, and a 4 B smaller payload due to serialized key names. No learned or neural-network capacity increase is established.

## Amendment disclosure

An independent review identified unsupported earlier accounting: query MAC count omitted the query axis, native/Mirror MAC formulas were asymmetric, the reported state byte count used float64 teacher arrays, reset testing did not reuse the same state object, and serialization was not reloaded. Amendment 01 corrects those implementation and accounting facts. Deterministic replay of the same seeds corrected metrics only; no tuning followed fresh access.
