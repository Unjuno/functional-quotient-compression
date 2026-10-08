# MA-369 status

- Status: **FAIL** (bounded OFA-style MLP screen)
- Branch: `research/ma-369-ofa-subnetwork-mirror-correction-20261008`
- Protocol frozen: before development; selected learning rate 0.01 on worlds 36900/36901
- Fresh worlds: 36910/36911/36912 evaluated once under frozen settings
- Results: 420 architecture/method/world/LR rows; complete actual torch.save payload bytes charged
- Verification: 3 unit tests pass; payload hashes and metrics replayed; exact inference roundtrip checked
- Next candidate: select the next UNTESTED P0 in live registry/queue (MA-369 now resolved)

## H — falsifiable hypothesis

Four per-subnetwork Givens angles over one shared OFA-style supernet recover width/depth sharing interference on held-out digits, staying within 2 accuracy points of independent subnetworks at <=50% their complete serialized bytes; Mirror should also beat simple same-budget correction controls.

## T — execution

Small two-layer slimmable MLP (input 64, max hidden width 64, ten classes), six width/depth subnets, sandwich-rule training with in-place distillation, 500 shared and independent updates, 100 correction updates per subnet. sklearn digits v1.8.0; stratified 60/20/20 splits; development worlds 36900/36901 selected LR 0.01 from {0.003,0.01}; fresh worlds 36910/36911/36912. Controls: uncorrected shared, four-angle Givens, four-value FiLM, rank-1 output LoRA, independent subnet bank. CPU PyTorch 2.14.1, one thread.

## D — decision

**FAIL** for the registered Mirror-specific correction hypothesis. Across fresh rows averaged over subnet architectures, accuracy was 95.09% Mirror, 95.00% shared, 95.09% FiLM, 95.52% LoRA, 95.71% independent. The Mirror gain over shared was 0.09 percentage points, below the 1pp gate, and FiLM matched it at equal payload bytes with fewer correction MACs. Mirror payload was 40,191 bytes (41.0% of the 98,185-byte independent bank); this storage gain comes from shared OFA weights and is not Mirror-specific. Mean NLL was worse for Mirror than FiLM and independent.

## C — strongest counter-hypothesis

The model's shared supernet already transfers sufficiently across nested widths/depths; when residual specialization helps, ordinary rank-1 LoRA or diagonal FiLM is at least as effective. The observed byte reduction is caused by subnet weight sharing, not the Mirror coordinate.

## U — unresolved

No full Once-for-All reproduction, larger language/vision task, near-convergence capacity study, hardware latency benchmark, or broader width/depth schedule. Fixed updates are learning-efficiency evidence only. The reported Python CPU throughput is noisy and does not establish device runtime.

## Evidence separation

- **Fact:** Fresh mean accuracies and full serialized byte counts are recorded in `RESULTS_CORE.csv`; 420 rows replay with max NLL difference 4.7e-9; all payload hashes match and serialization logits are exactly equal.
- **Interpretation:** This bounded task supports compact shared subnet deployment but does not support Mirror-specific value.
- **Hypothesis:** Shared width/depth supernets may benefit more from learned low-rank residual corrections than orthogonal coordinate views; test that only in a separately frozen experiment.
