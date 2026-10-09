# MA-429 status: FAIL (depth extrapolation improves; byte gate misses)

## H — falsifiable hypothesis

A continuous Givens depth view on one shared recurrent block would extrapolate beyond trained depths with <=90% of a linear timestep embedding's bytes and equal recurrence count.

## T — executed

Synthetic 2D recurrent trajectory with teacher transition Q(0.12d)AQᵀ, trained at depths 1–4 and evaluated at depths 1–8. Compared static shared matrix, continuous Mirror timestep, linear matrix embedding A+dB, and four discrete depth matrices held at the fourth matrix after training depth. Each model ran the same eight recurrence steps. 400 AdamW updates × batch 128; 2 development worlds × 3 seeds selected LR 0.01; 3 fresh worlds × 3 seeds. Spectral norms projected to <=0.95. Actual serialized inference payloads measured.

## D — FAIL by byte gate

Fresh mean normalized error over depths 5–8: shared 0.2436; Mirror 0.0133; linear time embedding 0.1279; discrete depth 0.1131. Mirror gives a strong aligned extrapolation improvement, with the same recurrence count and stable matrices. Its payload is 1,829B, exactly equal to linear-time embedding and larger than discrete depth (1,641B), missing the <=90% byte gate.

## C — strongest counter-hypothesis

The teacher is generated from a continuous Givens depth law, so the quality result is aligned feasibility. The 2x2 operator is too small for raw parameter savings to survive serialized tensor/key overhead; equal-byte outcomes dominate the storage claim.

## U — unresolved

Higher state dimensions, nonlinear recurrent blocks, realistic Transformer depth, language quality, and fused inference remain untested. MA-431's same-family composition question is deferred for redesign because the MA-424/425/429 sequence shows repeated actual-byte failure from tiny serialized operators.
