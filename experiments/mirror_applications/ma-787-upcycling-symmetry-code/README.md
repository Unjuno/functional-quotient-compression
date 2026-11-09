# MA-787 — Expert upcycling symmetry-breaking View code

Status: **FAIL** for Mirror-specific value; aligned-family learning-efficiency signal observed.

## Fact

Eight synthetic child experts start from one duplicated parent. At heterogeneity σ=0, 32 calibration examples/child plus a shared rank-1 code yield heldout-child nRMSE `5.94e-8` using 1,678 B and zero optimizer updates. Full continued pretraining reaches .00091 using 3,442 B, 3,200 updates and 61 ms CPU training; identical duplicates remain at .212 error. A same-size random perturbation code remains at .212. Mirror exactly aliases native rank-1 LoRA by payload hash and predictions. At σ=.05, Mirror error is .241, while full continued pretraining reaches .00304 with the same 3,200 updates.

## Interpretation

For the designed aligned child family, a 32-example scalar code adapts much more cheaply than full child updates. This is a fixed-budget adaptation-efficiency signal, not added capacity: the function is standard rank-1 LoRA. A single shared direction cannot accommodate the private variation at σ=.05, where continued pretraining succeeds and the code misses quality badly.

## H / T / D / C / U

- **H:** A compact child-specific View breaks duplicated-expert symmetry and recovers child behavior with fewer bytes and less adaptation compute than full continued pretraining.
- **T:** Seed 78701; eight 10×10 linear child tasks; six child identities fit the rank-1 basis, two identities held out for code adaptation using 32 examples; σ=0/.05; compare identical duplicates, random direction, Mirror/native rank-1 LoRA, full CPT (400 updates per child) and independent bank. Actual compressed NPZ bytes, heldout/all-child nRMSE, examples, updates and CPU training time recorded.
- **D:** **FAIL** for Mirror-specific value. Aligned children show a useful fixed-budget adaptation signal, but native LoRA exactly aliases it; σ=.05 requires private capacity and the code fails the quality gate.
- **C:** The positive world is generated from one shared rank-1 residual direction; the advantage is ordinary low-rank adaptation over full matrix updates.
- **U:** Trained MoE upcycling, natural expert tasks, router/loss recovery, larger heldout worlds, and the compute/storage Pareto under realistic GPU kernels.

## Scope

Synthetic linear task bank only; no pretrained MoE or continued pretraining corpus. CPT is full-matrix minibatch SGD from the duplicated parent, not a production Expert-Upcycling recipe. No capacity claim is made. Fresh data remained sealed after exact native LoRA alias.
