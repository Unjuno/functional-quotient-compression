# MA-569 status

- Status: **FAIL for Mirror-specificity; recurrent-step application beyond this mechanism NOT ESTABLISHED**
- Branch: `research/ma-569-universal-transformer-step-view-20261102`
- Protocol freeze: `dbbd1687`; A1 corrects MAC accounting only.
- Fresh worlds: 56911–56913; fit steps 1/4; heldout 2/3/5/6.

## H / T / D / C / U

**H:** A compact affine step-angle code on one recurrent block recovers unseen terminal-step functions at fewer bytes than independent maps; compare no-view, direct time-angle and static per-step rank-2 controls.

**T:** Shared 8-channel nonlinear recurrent transition; three fresh seeds; each seed calibrates two coefficients on steps 1 and 4, then evaluates 2,3,5,6. Zero optimizer updates. Methods include shared recurrence without view, Mirror, explicit direct time-angle, static per-step rank-2, and independent effective maps. State and code are serialized and reloaded.

**D — FAIL for Mirror-specificity.** Fresh mean NRMSE: native `0.0261`; Mirror/direct time-angle `8.54e-17`; static rank-2 `4.35e-17`; independent `7.50e-17`. Mirror and direct time-angle have identical output, actual total bytes (`2,395 B`) and active MACs (`82,944`) across every fresh heldout row. The direct control is an exact functional alias.

**C:** Teacher step angles are affine in terminal iteration, exactly the law stored by Mirror. The direct two-coefficient time-angle code uses the same law and is byte/compute/function identical.

**U:** Trained Universal Transformer, algorithmic generalization beyond the synthetic recurrence, language NLL, realistic time embeddings, accelerator latency/energy and fresh replication on learned weights.

## Facts

- Fresh mean/max heldout NRMSE: native `0.0261/0.0511`; Mirror/direct `8.54e-17/3.90e-16`; static rank-2 `4.35e-17/5.87e-17`; independent `7.50e-17/1.13e-16`.
- Mean serialized total bytes: native `2,184 B`; Mirror/direct `2,395 B`; static rank-2 `2,507 B`; independent `6,539 B`.
- A1 active matrix/view MAC proxy includes input projection, all recurrent steps, final projection and view: native/independent `98,304`; Mirror/direct/static `99,328` averaged over heldout step counts. CPU timings include full recurrent execution and output but are Python/NumPy specific.
- Direct alias outputs, metrics, bytes and hashes verified; zero optimizer updates.

## Interpretation

A two-value recurrent time code transfers a deliberately matching affine Givens law to heldout step counts with substantially fewer stored bytes than an independent transition per step. The direct time-angle control exactly reproduces it, so the result supports ordinary compact step conditioning in this synthetic family, not Mirror-specificity or a trained Universal Transformer claim.
