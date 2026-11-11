# MA-813 — code-only new skill

## H / T / D / C / U

**H:** A compact code on a frozen shared rank-2 action basis adds a few-shot skill with lower complete storage and adaptation work than fresh rank-2 LoRA.

**T:** Synthetic 8D linear policy; shared core, two frozen random rank-1 atoms, six base skill codes, one new skill generated from the same basis; infer coefficient from 32 support examples and score 512 heldout examples; seed 81301; no optimizer updates. Compared frozen core, Mirror code, native shared-basis code, fresh rank-2 LoRA fitted to the same support set, and independent matrix.

**D: FAIL.** Mirror code reaches heldout nRMSE 2.87e-8, but full serialized deployment is 1,440 B versus 981 B for fresh rank-2 LoRA and 754 B for an independent new-skill matrix. Native shared-basis code is functionally identical at 1,456 B. The core-only model is 719 B but misses quality (nRMSE .0481). Adaptation uses 512 scalar-response operations for both code and LoRA controls.

**C:** Charging the frozen shared basis and its deployment metadata makes a new code more expensive than storing a fresh small LoRA or matrix. The code is exactly ordinary shared-basis coefficient fitting.

**U:** A basis trained on natural motor skills, nonlinear policy adaptation, multiple skill amortization, robot rollouts and fresh seeds. Fresh remained sealed after byte and native-control failure.

## Evidence separation

- **Fact:** Serialized payload bytes, hashes, heldout metrics, and replay results are in `RESULTS_CORE.csv`.
- **Interpretation:** The frozen synthetic basis addresses the target function but does not beat ordinary LoRA or full-matrix storage after all deployed state is charged.
- **Hypothesis:** Code-only skill addition may become useful after enough skills amortize basis cost; this one-new-skill screen does not establish that.
