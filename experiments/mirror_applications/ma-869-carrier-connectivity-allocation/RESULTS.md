# MA-869 results: carrier connectivity and code allocation

## H — Hypothesis

A small observation address can choose the most reliable carrier that is connected to an aggregator, preserving near-central linear-Gaussian inference while reducing message-hop cost. Mirror-specific attribution additionally requires the address representation to outperform the native connectivity/reliability rule at comparable payload size.

## T — Trial

The frozen synthetic world used four independent unit-variance latent coordinates, eight private noisy observation carriers, and a line mesh with one possible cut. Each coordinate had one near and one far carrier; per-coordinate reliability assignment was seeded, with noise sigma 0.25 or 0.75. Fresh seeds 701, 719, and 733 each generated 4,096 draws. Controls were reachable-central posterior, best-reliability selection ignoring connectivity, connectivity-aware best reachable selection, random reachable selection, plus native-rule and Mirror-address payloads. The message wire format was uint8 agent ID + uint8 type ID + float32 observation (6 bytes), with every hop charged. Development seed 101 was used only to verify implementation; no tuning occurred.

## D — Decision

**FAIL for Mirror-specific attribution.** The connectivity-aware utility gate passed in all three fresh worlds, but its address table exactly encoded the same choices as the native rule and cost more bytes.

## C — Strongest counter-hypothesis

The result is explained by ordinary graph reachability plus reliability metadata and a native deterministic selection rule. A Mirror address is unnecessary; in this small fixed schema the rule is 3 bytes versus a 5-byte Mirror descriptor/address table.

## U — Unconfirmed

Arbitrary graphs, changing topologies, learned/noisy reliability classes, multiple simultaneous cuts, high-dimensional correlated latents, privacy leakage, and neural collective inference remain untested. The screen does not establish exact recovery for general Mesh Inference.

## Facts

- Across fresh seeds, mean normalized MSE was 0.08457 for connectivity-aware selection and 0.08088 for reachable-central collection; the largest per-world difference was 0.00607, below the frozen 0.02 gate.
- Connectivity-aware selection used 4/6.67 mean messages and 84/164 mean message-hop bytes, a 48.8% reduction in hop bytes. Maximum shortest-path latency averaged 5.67 hops versus central 6.67.
- Connectivity-aware selection had zero disconnected selections. The policy ignoring connectivity selected one unavailable far carrier on cut seed 733 and reached nMSE 0.29851 there.
- Random reachable selection averaged nMSE 0.15911 and 92 hop bytes.
- Charged schema was 14 bytes. Native schema+rule payload was 17 bytes; schema+Mirror descriptor+four addresses was 19 bytes. Native and Mirror choices were identical for all worlds.
- Each fresh world used 4,096 latent draws. The isolated script's maximum measured wall time was 0.0181 s; timings are host-specific and do not include a trained model or network hardware.

## Interpretation

Carrier connectivity is a meaningful structural gate: ignoring a cut can lose an entire coordinate, while selecting the best reachable carrier keeps the quality close to the reachable-central posterior at lower communication cost. This utility is fully accounted for by a compact native routing rule over paid topology/reliability metadata. Mirror's extra address state increases storage without changing decisions, so there is no Mirror-specific gain here.

## Hypothesis for follow-up

A future test could make topology and reliability vary per query and compare a shared rule program against explicitly stored Mirror addresses under the same dynamic-schema and message costs. That would be a new protocol and is not inferred from these results.
