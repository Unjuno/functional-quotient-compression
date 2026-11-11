# MA-848 — FAIL for Mirror-specific attribution

## H / T / D / C / U

- **H:** A persistent task coordinate composed with transient Hebbian episode state retains unseen task×memory combinations while using less storage than independent pair matrices.
- **T:** Synthetic D=16, 8 task gains × 6 episodes, 24 seen and 24 checkerboard-heldout pairs; one-shot 16-key support per episode. Development seed 17, fresh seeds 37 and 59. Controls: full effective matrix per pair, native task-code × fast-state factorization, Mirror task View × episode View. No optimizer updates; 96 support examples/world.
- **D:** **FAIL** for Mirror-specific value. Both factorized methods reach heldout nRMSE ≈2.5e-8 and exact reset (max leakage 0), using 6,760 B for native factorization and 6,764 B for Mirror versus 49,488 B for independent matrices (86.3% fewer serialized bytes). Mirror adds 4 serialized bytes and 6.25% query MACs over native factorization.
- **C:** Ordinary factorization of persistent task identity and transient fast state fully explains compositional recovery and storage savings.
- **U:** Learned task inference, nonlinear plasticity, noisy keys, long episodes, real training/runtime, and whether a learned Mirror code can outperform native factorization.

## Facts

Across both fresh worlds, mean heldout nRMSE is 2.52e-8 (seed 37) and 2.50e-8 (seed 59); maximum is 2.81e-8. Serialized payloads: 49,488 B full pair matrices; 6,760 B native factorized state; 6,764 B Mirror. Episode fast state is 12,288 B; Hebbian write proxy is 24,576 MACs. Query proxy is 6,144 MACs native and 6,528 Mirror. Fast state reset error is zero. The heldout pair outputs are computed compositionally from task gain and episode memory factors.

## Interpretation

This establishes a useful synthetic storage frontier for factorizing repeated task×episode effective matrices, but not a Mirror-specific gain: the ordinary native factorization has identical outputs and a 4 B smaller payload, with lower query compute. The result does not establish a learned or neural-network capacity increase. Fresh seeds remained untouched during protocol selection.
