# MA-342 — HyperLoRA generator outputs a Mirror code

Status: SCREENING. Prior art: PA38 HyperLoRA.

## H

On held-out clients whose LoRA updates lie in one shared rank-2 basis, generating a scalar phase code over shared basis factors may match HyperLoRA's client-conditioned full LoRA factors at lower inference bytes. A generic two-coefficient code may fully explain any Mirror gain.

## Mirror insertion

> **Mirror insertion:** this experiment adds a client phase `m` to shared LoRA factors so that a compact rank-2 task delta is produced without generating/storing per-client LoRA factors.

Compare shared base only, shared U/V factors plus phase Mirror code, the same factors plus direct two-coefficient control, a HyperLoRA-style hypernetwork generating per-client LoRA A/B factors, and independent per-client LoRA factors. Evaluate only the effective ΔW=A@B and predictions, not raw factor similarity. All hypernetwork, basis, client-code and metadata bytes are paid.

## T

Input/output dimension 8, LoRA rank 2, 16 training clients and 8 unseen midpoint clients; 64 training examples/client and 256 disjoint test examples/client. Teacher: W(phi)=W0+U diag(cos(phi),sin(phi)) Vᵀ. Development worlds 34221–34222, fresh 34231–34233. Train shared factors, coefficient factors and a 2→32→32 HyperLoRA analogue for 1,000 Adam updates; independent reference fits from held-out support examples. Report test MSE, effective-delta reconstruction, actual serialized bytes, client communication-code bytes and compute/time.

## Gates

**PROMISING:** Mirror reaches test MSE <=1e-3 and beats HyperLoRA payload by >=20%, while beating direct coefficients by >=10% at comparable quality. **FAIL:** a generic coefficient control is within 5% bytes and quality, or quality fails. No capacity claim from fixed updates.

## C / U

The teacher is a planted low-rank orbit favorable to the code. This does not reproduce HyperLoRA federation or establish natural adapter compression, training stability across tasks, privacy or communication rounds.


## Results

**H:** a phase code over shared LoRA factors could match a HyperLoRA-style generator at lower bytes, but direct coefficient controls may explain the gain.

**T:** three fresh worlds with 16 training clients and 8 unseen midpoint clients; 64 training and 256 test examples per client. Synthetic updates follow a shared rank-2 phase orbit. Mirror, direct coefficient factors, and a 2→32→32 HyperLoRA-style factor generator received 1,000 Adam updates. Independent rank-2 adapters were fit from held-out support examples. Effective task updates are evaluated as matrices/predictions, with no raw LoRA factor comparison.

**D — FAIL for Mirror-specific byte gate:** fresh mean held-out MSE: Mirror phase 516 B / 1.0014e-4; generic two coefficients 556 B / 1.0014e-4; HyperLoRA-style generator 5,048 B / 1.0298e-4; independent rank-2 client factors 2,411 B / 1.0552e-4; base-only 341 B / .003925. Mirror saves 89.8% against the hypernetwork, but only 7.2% against the generic control, below the 10% gate. Mirror/generic training time means were .460/.464 s; hypernetwork .574 s.

**C:** task updates are deliberately generated from the same two-direction phase family, making this an aligned feasibility test. The generator is a synthetic analogue, not a reproduction of HyperLoRA's federated training.

**U:** real client adapters, federated communication rounds, privacy, non-IID shifts and useful downstream task quality are untested.

### Fact / interpretation / hypothesis

**Fact:** 15 fresh rows; payload hashes and test MSE replay exactly (max difference 0); 3 tests pass.

**Interpretation:** the phase code reduces representation bytes relative to direct coefficients only slightly and is mathematically the same two-coefficient function family. The large delta versus full factor generation is shared-basis compression, not a Mirror-specific gain.

**Hypothesis:** natural adapter banks may need private residuals, and the shared/private boundary should be tested before an end-to-end federated claim.
