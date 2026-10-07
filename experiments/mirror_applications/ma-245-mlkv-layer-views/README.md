# MA-245 — MLKV shared cache + per-layer Mirror KV views

Status: SCREENING
Evidence lane: MECHANISM / STORAGE / RUNTIME
Base commit: `1c313c1ac6aba2b5c4cd4bcc3933c08b1f98ce19`

## H — hypothesis

When per-layer key/value roles are related by small orthogonal coordinate changes, one shared physical K/V cache plus per-layer Mirror addresses recovers layer-distinct attention outputs while storing fewer cache bytes than MLKV with multiple layer groups. The Mirror must also beat a same-size scalar-gate control.

## Prior-art delta

PA08, *MLKV: Multi-Layer Key-Value Heads for Memory Efficient Transformer Decoding*, shares KV heads across attention heads and across layers. It reports cache sizes proportional to `2 × batch × sequence × layer_groups × kv_heads × head_dim`, and tests uptrained Pythia variants. MA-245 tests whether layer-specific low-description Views can reduce the number of stored cross-layer cache groups while restoring layer-level output distinctions.

Controls: independent layer K/V projections (MHA upper control), two-group MLKV, one-group hard-tied MLKV, and a byte-near scalar layer/role gate.

## Synthetic mechanism

Four layer queries attend to the same eight-token memory. The teacher has one shared key projection and one shared value projection, each transformed by a layer-and-role Givens angle. Students predict the teacher's four attention outputs. This is an optimistic, coordinate-aligned feasibility screen; the shared memory input isolates projection/cache reuse without modeling evolving Transformer hidden states.

## Gates

PASS requires in every fresh world: Mirror MSE within 10% of independent-layer MHA, at least 10x lower than one-group hard MLKV, at least 10% lower than the byte-near gate, and at least 25% fewer serialized model bytes and materialized K/V cache bytes than two-group MLKV.

FAIL if any quality threshold is missed or the scalar gate is within 10% MSE at equal or lower bytes. NOT ESTABLISHED if fresh worlds, verification, or actual cache tensor accounting are incomplete.

## H / T / D / C / U

H is above. T, D, C, U and fact/interpretation/hypothesis separation will be completed after development-only selection, frozen source, and fresh replication.
