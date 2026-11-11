# MA-819 — shared equilibrium and system-role Views

## H / T / D / C / U

**H:** A shared equilibrium forecaster plus compact role Views improves multi-system rollout storage and error over independent system forecasters, beyond native system embeddings.

**T:** Synthetic stable linear system, dimension 8, eight system roles, horizon 16, seed 81901. Compared independent matrices, shared-only transition, native low-rank system embeddings and Mirror role codes. 64 initial states per role. Zero optimizer updates.

**D: FAIL for Mirror-specific attribution.** Independent forecasters use 2,790 B. Shared-only uses 1,113 B but heldout-scored systems have nRMSE .02077 (just above .02); role factorization uses 1,911 B with zero error. Native embedding and Mirror payloads have identical sizes, decoded transitions, stability and MAC proxy. The apparent 879 B saving versus independent is ordinary shared low-rank system embedding, not Mirror-specific.

**Evaluation boundary:** “Heldout systems” here means separately scored IDs only. Their role codes are present in both native and Mirror inference payloads; this run does not test unseen-system generalization or code inference. The seed used known synthetic transitions and no training.

**C:** Native system embeddings exactly account for the shared-state role specialization. Shared-only is cheaper but narrowly misses the declared rollout tolerance.

**U:** Actual ESE models, learned role codes, truly unseen systems, nonlinear dynamics, real forecasting data and fresh seeds. Fresh remained sealed.

## Evidence separation

- **Fact:** Full bytes, hashes, horizon errors, spectral radii, compute proxy and replay errors are in `RESULTS_CORE.csv`.
- **Interpretation:** Shared parameterization reduces storage versus independent models, but the Mirror insertion adds no function beyond native low-rank system embeddings.
- **Hypothesis:** A future useful result needs learned inference of role code for unseen systems and lower full bytes than native ESE conditioning.
