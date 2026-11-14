# A8 — Rank-certified limits of shared-feature linear heads and conditional nonlinear alternatives

Date: 2026-11-15. Status at writing: **PRE-RUN**, no A8 model training performed. Exploratory continuation of negative A1–A7 results; not an independent preregistration or population-level proof.

## Research question
If 16 target functions have sample response matrix rank exceeding the width of a task-independent shared embedding, can a low-byte task-conditioned nonlinear model improve function reconstruction? If yes, is this specific to a reversible shear Mirror, or also achievable with ordinary conditional shifts/FiLM/input affine transforms?

## Dataset
- Two **symbolically defined, non-Mirror** teacher families with input x in R: (a) shifted hinge `max(0,x-t_k)` and (b) interval ramp `max(0,x-t_k)-max(0,x-u_k)` with independently sampled right endpoint u_k>t_k.
- K=16 task IDs, each known during training and evaluation. Teacher parameters are sampled with fixed per-world seeds, teacher code does **not** use Mirror matrices. They are not accessible to students.
- Six worlds 810..815, two initialization seeds 0,1, two teacher families, serialized-byte caps 256 and 512 B, seven models = 336 training runs. Development-only world 809 may be used for smoke/timing and will be excluded from audit.
- Train/validation/IID test use separate uniform x samples from [-1.5,1.5]. OOD test x uniform [-2,2], separately sampled. Every task uses the same input set per split; task labels computed analytically. Within-world x splits are independent. All domain values dimensionless.
- Teacher t_k: sorted evenly spaced values in [-0.95,0.75] plus small independent jitter (maximum 0.02), ensuring unique thresholds. For ramp, u_k=t_k+d_k, d_k uniform [0.2,0.55]. The parameters are fixed by the world seed.
- N_train=1024, N_dev=384, N_iid=384, N_ood=384. Audit labels and samples are not used to choose hyperparameters or checkpoints.

## Models and training
- `mirror`: trainable shear Q around ReLU with exact inverse; `pre`: same but no inverse; `film`: gain + shift from a two-coordinate task code; `shift`: directly learned rank-two additive hidden shift; `input_affine`: task-specific input affine transform before common FFN; `basis`: shared features with rank-two linear output adapter; `multihead`: task-specific affine heads on shared features.
- Each model is one-hidden-layer scalar-output ReLU MLP, all learned scalars FP32 CPU, task embeddings trainable except multihead. For each cap choose the widest feasible hidden width (even for Mirror/pre). Count all task embeddings, output heads, tensor biases; header 32 B + payload 4 B/scalar + SHA-256 trailer 32 B. No hidden numerical tensors beyond the declared state_dict. Executable decoder code and teacher generator are common/unpriced, so byte results measure a weight-file proxy, not total MDL.
- AdamW lr=0.012, weight_decay=0, 600 updates, batch 256 sampled uniformly over tasks and train x, gradient clip=5.0. Same initial model seed per world+seed+kind and independent frozen batch seed. FP32 CPU deterministic algorithms with one numerical thread. Fixed final-step checkpoint, no dev-based early stopping.
- A **construction** (not trained) will show a width-2 Mirror/pre realization for hinge and width-4 for ramp; it is a feasibility witness, not evidence of optimization success or Mirror-specific novelty. Direct formula teacher codec (16 or 32 threshold scalars) is acknowledged as a strong reference but not part of neural training competition.

## H / T / D / C / U
- **H (testable):** In at least 5/6 worlds, Mirror beats the validation-selected non-Mirror neural control on both IID and OOD normalized RMSE at one fixed cap; any storage advantage also requires achieving NRMSE<=0.08 with >=10% fewer bytes than the least-byte successful control.
- **T:** 336 locked training checkpoints, then validation selection separately for each family/world/cap using average dev MSE over seeds, followed by independent IID/OOD audit on saved weights. Rank certificate computed for shared-head widths using SVD of centered target matrix on fixed probe points; compare the theoretical minimum error to empirical results.
- **D:** PASS only with above world and capacity gates; FAIL **for tested conditions** if no Mirror rate-quality improvement; UNCERTAIN for general models/real data. Preserve truncated quality-frontier cases as censored; never divide by unavailable minimum bytes.
- **C:** FiLM, shift, input-affine or low-rank output basis may have equal or better function efficiency; rank barrier concerns **linear heads**, not all shared nonlinear conditional networks. A model may fail due to finite optimization rather than function-class limits.
- **U:** Synthetic function family structurally simple and favorable to coordinate shifts; all six worlds share a function family, 2 training seeds and 600 updates; no matched active FLOPs or code-storage price; sample rank and singular-value truncation depend on float64 numerical tolerance; any 95% interval over world maps is exploratory, not domain-wide inference.

## Audit and statistical unit
- Before audit, lock SHA-256 for each complete FP32 serialized checkpoint and train manifest. Evaluation **reloads** serialized bytes. Record val/IID/OOD MSE, NRMSE normalized by taskwise variation across target values, and max residual. Validate all values finite.
- Per world/family/cap select strongest non-Mirror by lowest mean val MSE of 2 seeds. Report ratios of seed-averaged IID and OOD MSE against selected control, six-world geometric means, 6-world Student-t exploratory 95% intervals on log ratios. Code-based bootstrapping may be used as sensitivity, no selective inference claims.
- Audit RANK claim: for `multihead` width h, the centered N-by-16 predicted-output matrix has rank <=h. Compute best-possible centered rank-h squared Frobenius residual from SVD. Verify explicit numerical witness for each teacher; no claim about unrestricted encoders.

## Verification and publication
- Test-first assertions for serializer exact round trip + corruption rejection, widths under cap, explicit construction, numeric rank barrier, determinism and train/audit disjointness; reproduce at least four independent short runs before interpretation.
- Full raw training weights may be kept in a ZIP outside Git; commit source, tests, protocol, world-level and model-level aggregate tables, signed evidence hashes, report and README to the existing `Unjuno/functional-quotient-compression` repository on an isolated branch, then merge after verification.
