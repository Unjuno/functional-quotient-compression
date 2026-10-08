# MA-360 — Reversible shared block with Mirror depth views

Status: **FAIL for Mirror-specific value; reversible tied-block memory result is narrow synthetic evidence.**
Branch: `research/ma-360-reversible-mirror-block-20261008`  
Base: `c935a90`  
Prior art: PA48 reversible PEFT.

## H — Hypothesis

A scalar Mirror coordinate per application of one shared reversible additive-coupling block may preserve a multi-step task while reducing serialized weights and activation memory, with recomputation below 2x.

> **Mirror insertion:** this experiment adds a scalar coordinate `m_l` to each repeated application of one shared reversible coupling block so logical depth steps can differ without separate physical block weights.

Direct layer gates are the closest non-Mirror control.

## T — Protocol and amendment

Synthetic 32D regression from an untied additive-coupling teacher; depth 8, batch 64, 300 AdamW updates. Development seeds 36001/36002; fresh 36011/36012 locked. Compared untied, ordinary tied, tied with direct gates, tied with Mirror, and reversible tied controls. Metrics include actual serialized bytes, task MSE, saved activation bytes excluding parameter storage, MAC proxy, and train/one-pass wall times.

The original development payloads did not replay in the current container. Amendment A1 preserved them under `protocol_variants/pre_environment_replay_audit/`, fixed CPU threads at one, recorded runtime versions, and reran development seeds only. No model or gate was tuned; fresh seeds remain sealed. Ten amended rows replay exactly.

## D — Development result

**FAIL for Mirror-specific value.** Direct and Mirror had identical payload/hash/task MSE in both seeds: 2,600B at MSE 2.0094 and 2,606B at MSE 1.9862. Their saved activation count was also identical (204,896B/batch), so no Mirror-specific compression or memory effect appears.

The reversible tied block used 2,382B/2,400B and saved 81,956B/batch versus 139,264B for ordinary tied execution (41.1% fewer saved activation bytes). Its task MSE matched tied at 2.0364/2.0022, within 5% of the untied upper control. Training was 1.225s/1.003s versus tied 0.869s/0.759s (1.41x/1.32x), below the 2x ceiling. This is a narrow reversible execution point, not a Mirror benefit.

## C — Counter-hypothesis

Direct scalar gates exactly reproduce the Mirror view, so any apparent difference is attributable to shared/tied or reversible execution. The synthetic shallow CPU task may not expose natural Transformer activation-memory behavior.

## U — Unconfirmed

Fresh seeds, GPU scaling, long sequence training, Transformer quality, and optimized reversible kernels remain untested.

## Fact / interpretation / hypothesis

- **Fact:** see `RESULTS_CORE.csv`, `VERIFICATION.json`, and preserved pre-amendment data. Ten amended rows replay exactly.
- **Interpretation:** reversible tying reduced measured saved activation bytes but cost roughly 1.3–1.4x training time. Mirror matched a simple direct gate exactly.
- **Hypothesis:** a Mirror-specific gain would require a coordinate richer than direct scalar gating and must be separately tested against reversible native controls.
