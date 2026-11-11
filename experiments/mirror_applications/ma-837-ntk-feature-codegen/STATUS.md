# MA-837 status

Status: **FAIL** — scoped synthetic feature-generator screen. Fresh data remained sealed after an exact native conditioning alias.

## H / T / D / C / U

- **H:** A small generated Mirror coordinate could preserve quality while improving on native no-task-storage NTK feature generation in complete bytes or inference compute.
- **T:** One synthetic development seed (83701); 12 seen and 8 unseen task descriptors, descriptor dimension 2, feature dimension 24, 48 support examples/task, 128 heldout inputs/task; closed-form shared generator fit; zero optimizer updates. Compared native no-task-storage generator, direct conditioning, generated Mirror coordinate, cached code table, independent readouts, and shared mean. Focused test: 1 passed.
- **D:** FAIL for Mirror-specific value. Native generator, direct conditioning and Mirror-generated coordinate use identical 630 B payloads (identical SHA-256), identical seen/unseen nRMSE (4.30e-15 / 3.83e-15), and the same 36,864 forward MAC proxy. Caching codes raises payload to 936 B and adds 576 decode MACs with no quality change. Independent readouts use 2,062 B for seen tasks only; shared mean uses 523 B but nRMSE .973. Fresh seeds remained sealed.
- **C:** Direct descriptor-conditioned NTK feature generation already supplies the same task-conditioned function without storing per-task parameters. Cached Mirror coordinates add state and decode work without improving on that native baseline.
- **U:** No trained NTK-CL, PEFT, raw-support task-code inference, real continual-learning sequence, or fresh-seed evidence. Unseen descriptors test interpolation within the known linear task family only.

## Fact / Interpretation / Hypothesis

- **Fact:** The no-task-storage native generator and generated Mirror coordinate are byte-for-byte identical and produce identical seen and unseen predictions.
- **Interpretation:** Naming the task descriptor as `m` adds no freedom beyond ordinary input conditioning in this setup. Persisting codes weakens the storage/compute frontier versus the no-storage control.
- **Hypothesis:** A Mirror-specific advantage would require a code generator that changes the achievable function/quality frontier beyond native descriptor conditioning; this synthetic linear generator does not show one.
