# MA-367/368 family diagnostic: post-output width/depth codes

## Facts

- MA-367 tested one width-specific Givens correction after ReLU in a nested-width MLP. On two development seeds, it failed the narrow-width improvement gate, tied a direct scalar-gain control at 4,179B, used 5.6% more nominal MACs than US-Net and trained about 1.4× slower. Fresh remained sealed.
- MA-368 tested additive width and depth Givens coordinates after the final active hidden block in a shared elastic supernet. On two development seeds, held-out NLL was worse than the plain shared supernet and the byte-matched factorized scalar control. Mirror used 12,645B vs 12,161B for the supernet, and trained about 1.4× slower. Fresh remained sealed.
- In both studies, the insertion is after width/depth truncation and after the last nonlinear block. The matched ordinary controls capture the same address freedom at comparable or lower cost.

## Interpretation and action

These consecutive FAILs share a structural limitation: an output-side low-dimensional correction does not reconstruct features lost by nested width/depth truncation. The width/depth post-output-code family is paused. Do not treat MA-369, MA-371, or MA-372 as runnable under the same insertion without a redesign.

A new protocol may resume the family if it moves `m` inside repeated blocks or changes the sampled-training objective, and compares against an equally sized ordinary scalar/FiLM control. Fresh data from the failed screens remain sealed.

## Hypothesis

An intra-block width/depth coordinate or a training objective that exposes held-out truncation errors may recover useful configuration behavior; that is untested and requires a new amendment or MA with fresh seeds.
