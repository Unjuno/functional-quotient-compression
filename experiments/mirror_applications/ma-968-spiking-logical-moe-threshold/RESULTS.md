# MA-968 results — one spiking expert with logical threshold specialists

Status: **FAIL for the preregistered Mirror-specific and byte gates**. The aligned task demonstrates that threshold specialists can reuse a synapse vector; the direct threshold table is cheaper than the proposed Mirror code at this model scale. Generic task-specific synapses require substantially more private state.

## H — hypothesis

A shared spiking synapse vector plus a small task-role threshold coordinate will recover independent-expert quality on synapse-aligned tasks at lower actual payload bytes. On generic tasks, threshold-only views will fail and rank-limited private synapse residuals will reveal where sharing stops working.

## T — execution

**Fact.** A single-step, 16-input hard-threshold spiking unit was trained for eight observed roles on two synthetic teacher families. Each world contained 4,096 train, 1,024 development, and 2,048 fresh examples per role. The structured family shared one teacher synaptic vector and varied role thresholds. The generic family used independent teacher vectors per role. Development world was 96801; fresh worlds were 96811, 96813, and 96817. AdamW used sigmoid surrogate logits during training and hard threshold spikes for evaluation, with up to 300 full-batch updates and development early stopping.

**Fact.** Controls were one shared vector/threshold; shared vector plus direct per-role thresholds; shared vector plus Mirror `base + scale * m_role`; shared vector plus rank-2 private synapse residual; and eight independent experts with deterministic top-1 routing by the observed role ID. Every method received the same role identity. The role-to-expert routing map and metadata were serialized and charged.

**Fact.** Actual payloads are serialized `torch.save` inference state plus routing and model metadata. Fresh scores use the reloaded payload. The Mirror threshold code was decoded into the direct threshold vector and compared on fresh-world inputs; maximum logit difference was **0.0** in all six family×world checks.

## Fresh results

| Teacher family | Condition | Mean fresh accuracy ± world SD | Payload bytes | Active synapse MACs/example | Mean spike rate |
|---|---|---:|---:|---:|---:|
| Shared synapse, role thresholds | Shared | 82.87% ± 0.28 | 2,149 | 16 | 0.505 |
| Shared synapse, role thresholds | Direct threshold bank | 99.55% ± 0.05 | 2,149 | 16 | 0.501 |
| Shared synapse, role thresholds | Mirror threshold code | 99.53% ± 0.29 | 2,653 | 16 | 0.501 |
| Shared synapse, role thresholds | Rank-2 private synapse | 99.48% ± 0.05 | 2,781 | 80 | 0.501 |
| Shared synapse, role thresholds | Independent routed experts | 99.34% ± 0.12 | 2,597 | 16 | 0.501 |
| Generic task synapses | Shared | 63.05% ± 2.62 | 2,149 | 16 | 0.495 |
| Generic task synapses | Direct threshold bank | 66.16% ± 1.97 | 2,149 | 16 | 0.489 |
| Generic task synapses | Mirror threshold code | 65.48% ± 1.75 | 2,653 | 16 | 0.477 |
| Generic task synapses | Rank-2 private synapse | 80.10% ± 2.24 | 2,781 | 80 | 0.499 |
| Generic task synapses | Independent routed experts | 99.24% ± 0.11 | 2,597 | 16 | 0.490 |

The direct threshold bank has eight learned threshold values. The Mirror representation has a shared base, scale, and eight role coordinates. The direct threshold and shared scalar-threshold payloads both serialize to 2,149 B in the measured format despite their different tensor shapes. Mirror costs 2,653 B, 2.2% more than the independent upper control and 23.4% more than the direct threshold bank. The direct bank saves 17.3% versus independent, below its 20% gate.

## Compute and activity

**Fact.** Each example activates one routed role. Shared threshold, direct threshold, Mirror threshold, and independent expert inference each perform 16 synapse MACs and one threshold lookup. Mirror additionally evaluates its scalar base/scale/code expression (two scalar arithmetic operations). The rank-2 residual adds 64 MACs per example, reaching 80. Mean output spike rates are near 0.5 in both task families. These are simulator counts, not measured energy.

## D — decision

**FAIL.** On the structured family, the Mirror code is within 1 point of independent-expert quality but does not meet the <=80% byte gate: it is larger than independent experts. It also fails to improve by 2 points over the direct threshold-bank control, and its decoded function is exactly the native threshold table. On generic tasks, threshold-only views score 65.48%, versus 99.24% for independent experts. Rank-2 private synapses recover only to 80.10% while using more bytes and five times the synapse-MAC proxy.

## C — strongest counter-hypothesis

**Interpretation.** The teacher was deliberately simple and aligned: one scalar decision boundary per role. The direct threshold table already captures all useful role variation and is a stronger, cheaper control than the Mirror factorization. Generic teachers were independently generated, so rank-2 residuals are intentionally capacity-limited; higher-rank private synapses may restore quality, but would increase stored state.

## U — unresolved

- Multi-step recurrent SNNs with membrane-state and temporal threshold views.
- Learned routing when task identity is not supplied.
- Larger neuron pools where threshold storage is no longer dominated by archive metadata.
- Natural event streams, continual retention, and measured hardware energy/latency.
- Whether a different Mirror coordinate beats a native threshold bank or shared-basis controller.

## Evidence separation

- **Fact:** listed accuracies, payload bytes, MAC proxies, spike rates, seeds, and zero-logit alias checks.
- **Interpretation:** thresholds can specialize an aligned shared synapse pool, but direct native thresholds are sufficient; generic synaptic differences need private state beyond rank 2.
- **Hypothesis:** more structured temporal or population-level codes may yield a Pareto gain in larger SNNs; MA-968 does not demonstrate it.
