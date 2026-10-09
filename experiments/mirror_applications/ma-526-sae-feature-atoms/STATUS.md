# MA-526 status

- Status: **FAIL**
- Branch: `research/ma-526-sae-feature-mirror-atoms-20261009`
- Protocol frozen before development; Amendment 1 fixed safe SAE object loading before any metrics were generated.
- Development seeds 52601/52602: complete; deterministic replay exact.
- Fresh seeds 52611–52613: sealed and unopened.
- Registry/status board: FAIL after reconciliation.

## H / T / D / C / U

- **H:** a shared 64-atom SAE pool with eight coefficients would preserve explicit-FV quality, compress incremental code state, and beat native top-eight and global OMP.
- **T:** pinned Pythia-70m plus pinned 4x layer-3 SAE; 12 tasks choose the pool and four are held out; two seeds; 64 atom IDs and eight pursuit coefficients.
- **D:** FAIL. Mirror loses 1.463/1.567 gold-logprob nats vs explicit, misses accuracy tolerance on seed one, misses the global OMP margin on seed two, and uses more incremental bytes than OMP. The SAE makes standalone deployment 4,172,903 B larger than explicit FV deployment.
- **C:** the SAE's fixed dictionary is poorly aligned to the task function vectors; native global sparse coding is simpler and smaller.
- **U:** other SAEs, feature transforms, task families and broader language behavior remain untested.

## Evidence classification

- **Facts:** three tests pass; eight payloads, metrics, task splits and pool-selection audit replay exactly. Preflight failure is preserved; fresh is unopened.
- **Interpretation:** a small code bank does not translate to a smaller full system when its required SAE basis is charged.
- **Hypothesis:** an SAE-space transformation may help, as separately proposed by MA-527.
