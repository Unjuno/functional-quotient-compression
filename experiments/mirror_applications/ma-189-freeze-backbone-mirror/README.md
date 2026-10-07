# MA-189 — frozen backbone with two-sided Mirror task views

Status: SCREENING  
Evidence lane: continual adaptation, storage, sample efficiency, runtime  
Branch: `research/ma-189-freeze-backbone-mirror-20261007`  
Base commit: `8d0763ddc241667e6e5c0fdefad16b2d71152538`

## H — falsifiable hypothesis

Given a frozen shared linear backbone, one input-view angle plus one output-view angle can recover a new task in the two-sided Givens family with rank-2 LoRA-level quality at 64 examples and no more than one quarter of its incremental inference bytes. An unrelated task map should demonstrate where private parameters become necessary.

## Prior-art delta

PA05's *Infinite-Parameter LLMs* generates low-rank modulations from live data using a compact hypernetwork. Neural Subspace Reallocation stores/retrieves task LoRAs; it is a private-state retrieval alternative. PA06's Ouroboros uses input-conditioned LoRA modulation over a reused block. These establish generated adapters and compact task state as existing controls. This experiment asks whether a two-scalar structured address can match a generated low-rank task update for a deliberately specified transformation family while using fewer inference bytes and examples.

MA-186 tested sequential retention with one input-pair angle and found a large aligned byte benefit but missed its strict relative-quality limit on 2/3 fresh worlds. MA-189 changes the question: one independently acquired skill, both input and output views, and a fixed sample-efficiency curve. It does not reuse MA-186 fresh seeds or tune from its outcomes.

## T — protocol

One 16D-to-8D Task-0 map is the frozen shared backbone. For aligned worlds, Task 1 applies one Givens transform to the first input pair and another to the first output pair. Unrelated worlds use an independently sampled task matrix. The Task-0 function is measured before/after.

For each new task, methods train for 200 full-batch AdamW updates at nested budgets of 16, 64 and 256 examples. Controls are hard tying, the MA-186 one-sided view, rank-1 and rank-2 LoRA, a shared rank-2 hypernetwork with nonzero factor initialization, and independent full weights. Development seeds 18901/18902 compare LR .003/.01; fresh seeds 18911–18913 remain sealed until the registered gate permits access.

The shared backbone is treated as a pretrained physical object and charged in each inference payload. Inference records charge every tensor, task ID and codec header. Resume payloads separately include optimizer state. Every method and sample-budget point reports held-out MSE, actual bytes, examples, updates, active MAC proxy, wall time and throughput.

## Gates

At 64 examples, all three fresh aligned worlds must meet <=1.10x rank-2 LoRA held-out MSE, <=0.25x LoRA incremental inference bytes, and <=0.01 absolute Task-0 MSE increase. The 16/64/256-example curve is a secondary sample-efficiency result. If rank-1 LoRA or the shared hypernetwork matches at comparable bytes and compute, no Mirror-specific advantage will be claimed.

The experiment measures fixed-budget adaptation only. It cannot establish converged capacity or natural-task performance.

## Results

Development selected LR 0.01 by the preregistered mean Task-1 MSE at 64 examples across methods and both task families (0.6240 vs 0.6548 for LR 0.003). At 64 aligned examples, two-sided Mirror MSE was 1.08e-10 on average, versus 1.00e-2 for rank-2 LoRA; both fresh testing and the full 16/64/256 sample curves are still pending. For unrelated tasks, two-sided Mirror MSE averaged 1.74 while rank-2 LoRA averaged 0.94, showing the intended private-capacity boundary. No fresh result has been opened.
