# MA-530 status

- Status: SCREENING
- Branch: `research/ma-530-sae-feature-experts-20261009`
- PA102 read; router, experts, controls and gates frozen before implementation/development.
- Protocol SHA-256 stored in `freeze.json`.
- Development seeds 53001/53002 not yet run; fresh 53011–53013 locked.

## H / T / D / C / U

- **H:** a support-trained router selecting shared-pool SAE feature experts can approach the same-router independent FV bank with smaller actual total router+expert state.
- **T:** pinned Pythia-70m+SAE; 16 relation tasks; 8 support and 8 query examples each; standardized affine input router; residual-selected pool64 and OMP16 behavior experts; learned/oracle FV and global sparse controls.
- **D:** screening; no development metrics accessed.
- **C:** task identity may not be recoverable from the query input alone, and SAE-coded experts may lose too much quality or fail to amortize pool/router/SAE costs.
- **U:** route accuracy, routed quality, full serialized bytes, active compute, and deterministic replay remain unknown.
