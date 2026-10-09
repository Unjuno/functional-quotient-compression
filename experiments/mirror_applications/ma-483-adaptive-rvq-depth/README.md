# MA-483 — Adaptive number of Mirror codebooks

Status: **FAIL** for the preregistered byte gate.

## H / T / D / C / U

**H:** A development-selected residual stopping rule should use fewer stages and reduce actual serialized bytes while preserving fixed-rate reconstruction quality.

**T:** Synthetic 16D vectors with 1, 2, or 4 occupied orthogonal phase planes, K=32. Development worlds 48300-48301 selected threshold 0.0 under the preregistered NRMSE <=0.08, minimum-bytes rule. Fresh worlds 48310-48312 × seeds 0-2 compared fixed four-stage phase codes (including an explicit no-op token for absent stages) and variable-length adaptive codes. The adaptive payload stores a depth per function and only active stage indices. All tensors and metadata are charged in serialized `torch.save` bytes.

**D:** FAIL. Across 9 fresh banks, both methods had mean NRMSE 0.05615. Adaptive mean depth was 2.32 versus 4.0, but actual payload averaged 4,637B versus 4,829B (96.0%), missing the <=80% byte gate. The registered threshold grid [0,.01,.03,.06,.1] was below the smallest nonzero amplitude (.125), so development selected the first tied value and threshold sensitivity was not established.

**C:** Depth metadata and tensor/container overhead dominate savings at only 128 functions; this small synthetic bank does not amortize variable-length coding. The predictable sparse structure may also be represented more cheaply by static metadata.

**U:** Larger banks, entropy-coded lengths, learned/nonorthogonal residuals, and native learned RVQ adaptive stopping were not measured. Thus no general adaptive-depth conclusion follows.

## Reproduction

```bash
python experiments/mirror_applications/ma-483-adaptive-rvq-depth/source/run.py --phase development
python experiments/mirror_applications/ma-483-adaptive-rvq-depth/source/run.py --phase fresh
python -m unittest discover -s experiments/mirror_applications/ma-483-adaptive-rvq-depth/tests
```
