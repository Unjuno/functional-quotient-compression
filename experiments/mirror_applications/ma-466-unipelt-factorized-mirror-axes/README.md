# MA-466 — UniPELT components as factorized Mirror axes

Status: SCREENING  
Evidence lane: MECHANISM/STORAGE/QUALITY/COMPUTE  
Branch: `research/ma-466-unipelt-factorized-mirror-axes-20261008`  
Base commit: `7cf891f`  
Prior art: PA84 (UniPELT)

## H — Hypothesis

A rank-two factorization over task, layer and PEFT component can generate gates for 128 logical adaptations with low output error and smaller actual payload than a full UniPELT gate table. Component ablations test whether LoRA, bottleneck adapter and prefix axes each contribute useful behavior.

## T — Frozen setup

The task has 16 task IDs, 8 layer positions and 3 shared component functions: rank-one LoRA-like linear update, two-wide nonlinear adapter and prefix-like layer bias. Targets use task×layer×component logits generated from a seeded rank-two CP factorization. The Mirror model learns rank-two task/layer/component factors; the UniPELT control learns all 384 gate logits directly. Both train for 2,500 Adam updates on 32-example batches and are evaluated on 64 heldout inputs for each of 128 contexts. Development seeds are 46601 and 46602; fresh seeds 46611–46613 stay sealed. Every shared component tensor and gate/factor is included in actual `.npz` inference bytes.

## Prior-art delta

PA84 establishes learned gates over multiple PEFT components. This screen asks whether factorized Mirror axes reduce gate state while preserving the composed function. A direct native rank-two CP gate model tests whether the factorization has value beyond ordinary low-rank gate sharing. Each component is also ablated individually.

## Results

Pending frozen development runs.
