# MA-419 status

Status: **FAIL — verified development screen; fresh seeds sealed.**

## H
Periodic activation modulation from a compact code will yield zero-shot held-out functions at lower total payload than a generic latent-concatenation model.

## T
Synthetic four-component periodic signals; 256 training and 64 held-out codes; two development seeds. Controls include generic concatenation, native direct harmonic conditioning, support-adapted private harmonics, and oracle coefficients.

## D
The periodic activation model met the held-out quality/bytes/throughput thresholds against generic concatenation in both development seeds, but the exact native direct-harmonic control reproduced the same predictions and payload size. Under the frozen rule, that means no incremental Mirror-specific benefit and a FAIL. Fresh seeds were not accessed.

## C
The teacher was generated precisely as a smooth code-to-amplitude/frequency/phase map over four harmonics. The Mirror model hard-codes that matching function family, while the generic tanh MLP does not have the same periodic inductive bias. Its large advantage may therefore be task alignment, and the native harmonic conditioner is the same model algebraically.

## U
Natural signals, off-family frequency coverage, convergence of private harmonic support adaptation, and optimized/fused periodic serving are untested. The arithmetic proxy does not price transcendental sine cost; steady-state CPU throughput is reported separately. This synthetic result establishes neither broad capacity nor a Mirror-specific gain.

## Evidence separation
- **Fact:** two development worlds; exact native output alias; measured serialized bytes, NRMSE, and throughput replayed from stored payloads; fresh untouched.
- **Interpretation:** the harmonic parameterization has a strong aligned synthetic Pareto point over generic latent concatenation, but fails the Mirror-specific gate.
- **Hypothesis:** a natural periodic signal domain may benefit from this inductive bias; that requires native-domain tests and optimized runtime.
