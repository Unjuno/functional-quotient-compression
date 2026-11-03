# MA-592 — Prompt pool + Mirror composition

Status: SCREENING  
Branch: `research/ma-592-prompt-pool-mirror-composition-20261009`  
Base commit: `e6e7ddc8`

## Hypothesis

**H:** Query-conditioned rank-2 coordinates over an eight-entry soft-prompt pool can preserve per-task quality while improving on native L2P top-1 selection without storing private prompts per query.

PA119 establishes that prompt pools with query-key routing are native controls. MA-591 prompt banks are reused as fixed pool entries. Query keys come from frozen Pythia hidden states; `m` softmax-weights the top two entries. Native top-1 and native top-2 weighted composition are explicit controls.

## Scope

This is a small language prompt routing screen over eight WikiText article tasks per seed. It does not establish class-incremental vision retention or a universal L2P replacement. Charge prompt pool, keys, metadata, virtual tokens and routing/mixture compute.
