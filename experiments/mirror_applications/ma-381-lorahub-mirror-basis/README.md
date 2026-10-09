# MA-381 — LoRAHub over a Mirror-compressed candidate basis

Status: **FAIL on the frozen development gate; fresh seeds remain sealed.** Dedicated branch: `research/ma-381-lorahub-mirror-basis-20261009`.

## Mirror insertion

**Mirror insertion:** this experiment adds one Givens angle `m_i` inside each shared rank-2 LoRA basis so that eight candidate update functions can vary by source task without storing eight independent pairs of LoRA factors.

## H — Hypothesis

A per-candidate Givens view can preserve LoRAHub few-shot signed-composition quality at no more than 60% of independent candidate-bank payload, while scalar source gates cannot recover that quality.

## T — What was run

PA55/LoraHub freezes candidate LoRAs and fits signed scalar coefficients on few-shot target support. The target-side coefficient fit was identical across bank variants. Eight rank-2 source candidates were trained from 4,096 16D source examples, then four target compositions were adapted using 64 support examples each (300 Adam updates; LR .01). Compared independent LoRA modules, hard tying, scalar gates, unrestricted 2×2 shared-basis coefficients, and one SO(2) Mirror angle per candidate. Development seeds were 38101/38102 under the protocol frozen at `4a20efd1`. Every source factor, code, target coefficient, dimension tag and NPZ overhead was charged.

## D — Decision

**Fact:** Independent bank source/test composition NRMSE was about 1.35e-7/1.87e-7 mean. Mirror achieved 2.79e-7 source and 2.06e-7 target-test NRMSE; generic 2×2 coefficients achieved 1.71e-7 and 1.95e-7. Scalar gates had target-test NRMSE .712/.537 by world; hard tying .682/.711. Mirror actual total payload was 2,396B in each world versus independent 3,935/3,922B (60.9%/61.1%), just above the frozen ≤60% gate. Generic coefficients used 2,497/2,500B at matched quality; scalar gates used 2,355/2,362B at much worse quality. The non-orthogonal LoRA factor-gauge audit changed full updates/predictions by at most 3.6e-15. Ten payloads pass byte/hash/metric replay; three tests pass.

**Interpretation:** In this aligned orbit, Mirror gives a large target-quality recovery over scalar source gates for 1.6% more total bytes and saves ~4.1% against the generic coefficient bank at similar quality. It also lowers source-bank MAC proxy from 512 (independent) to 352, equal to generic coefficients; composition summation costs 128 MAC per target example. Mean total training time was .81s Mirror, 1.48s independent, .74s generic coefficients on this CPU. The preregistered 40% payload saving was missed by roughly one percentage point, so the official decision is FAIL and fresh is sealed.

## C — Strongest counter-hypothesis

The source bank was intentionally generated on the same shared SO(2) orbit as the Mirror code. This is a favorable feasibility fixture. Fixed shared factors, few-shot coefficient vectors and archive metadata dominate at only eight candidates; larger banks might amortize them, while natural learned updates may not lie on one orbit.

## U — Unresolved

No pretrained language model, natural LoRAHub task bank, task-disjoint adapter generalization, larger candidate bank, or production latency was measured. Fresh worlds 38111–38113 were not opened because the total-payload gate failed. This is not a natural-task or capacity result.

## Evidence labels

- **Fact:** `RESULTS_CORE.csv`, per-world JSON and the 10 actual NPZ payloads record source and composition quality, bytes, MAC proxies, timing and hashes.
- **Interpretation:** Small Mirror codes recover logical variation missed by scalar gates, but the exact total storage threshold remains unmet at this bank size; generic low-rank coefficients are close in quality.
- **Hypothesis:** A larger candidate bank could amortize shared state and make the aligned quality/byte point pass; a new frozen protocol must test this separately and include task-identity-disjoint data.
