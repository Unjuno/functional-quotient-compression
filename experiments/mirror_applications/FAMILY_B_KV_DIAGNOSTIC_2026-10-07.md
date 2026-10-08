# Family B diagnostic — KV role/cache views (2026-10-07)

## Trigger

MA-061 and MA-063 were consecutive P0 KV candidates and both failed for the same demonstrated structural reason. Per queue stop rule, the current KV-view subfamily should pause for redesign; attention-head candidate MA-041 remains separate evidence.

## Facts

- **MA-061, one physical KV head to logical views:** Mirror and MQA both used exactly 256B cache per sequence. MQA had lower model payload (4,893B vs 5,333B), equal active-compute proxy, better output MSE at both development LRs, and higher measured CPU throughput.
- **MA-063, causal Mirror-MQA:** MQA and Mirror both used exactly 512B cache per sequence. MQA had lower model payload (4,893B vs 5,333B), equal compute proxy, better output MSE at both development LRs, and higher measured CPU throughput. Vectorized rotations did not alter the comparison.
- In both experiments, all bytes include serialized view coordinates/metadata. Neither run opened fresh worlds after its predeclared development screen showed simple MQA dominance. Replay and tests passed.

## Interpretation

A coordinate applied at read time does not reduce persistent KV state if MQA already stores one physical K/V stream. It adds model bytes and transform work. The tested view parametrizations also failed to recover the aligned teacher within the selected fixed-update budget.

## Redesign gate before reopening Family B

Future KV candidates should identify a physical cache reduction beyond MQA, for example lower-bit cache values, a smaller latent cache plus deterministic reconstruction, or fewer bytes read per decode step. Protocols must separately charge reconstruction state and measure actual cache tensor bytes; an expanded logical-head count alone is not evidence of capacity.

## Handoff

Continue the global P0 queue at Family C, starting MA-076. MA-041's aligned aggregate-attention result remains PROMISING but does not override this KV-specific diagnosis.
