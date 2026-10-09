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
