# MA-260 — BatchEnsemble rank-one Mirror ensemble

**H:** structured orthogonal views improve four-member ensemble quality per byte over native BatchEnsemble.

**T:** synthetic four-class Gaussian-mixture task, shared 2-layer MLP, native rank-one factors, Givens views, single and independent controls; fresh worlds 26010–26012 × seeds 0–2. Charge actual serialized payloads.

**D:** pending.

**C:** rank-one member scaling may match Mirror diversity with less compute.

**U:** quality, bytes, calibration and throughput.

## Fresh outcome and validity

Exploratory averages from the frozen runner: BatchEnsemble 0.237 accuracy / 3.567 NLL / 0.486 ECE / 10,453 B; Mirror 0.232 / 3.375 / 0.547 / 10,261 B; single 0.230 / 3.383 / 0.546 / 8,797 B. The runner inspection found it trains every evaluation on world 26000, even for fresh worlds 26010–26012. Thus these metrics do not test fresh-world generalization and the registered claim is NOT ESTABLISHED. Raw files and hashes are preserved; no fresh-informed changes were tested.
