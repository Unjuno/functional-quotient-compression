# MA-736 — Shared operator plus sparse private physics residual

Status: FAIL (scoped synthetic Mirror-specific claim)

## Fact

The independent eight-regime bank is 12,894 B. On the two held-out diffusion regimes, a shared mean is 2,063 B with global/interface solution nRMSE `.03352/.02634`. The rank-4 native PCA and Mirror banks are both 9,140 B with `.02067/.02240`. Adding top-8 private residual entries per regime reduces errors to `.01882/.01879` at 10,226 B; top-64 reaches `.01342/.00985` at 11,219 B. Thus the first tested private setting meeting both `.02` quality limits saves only 20.7% against independent, short of the 25% gate. Mirror and native PCA/private controls match payload hashes, bytes, outputs and multiply counts exactly. Full table: `RESULTS_CORE.csv`.

## Interpretation

The synthetic variable-diffusion operator family admits shared compression, but the held-out interface regimes need paid private state to cross the quality threshold. In this screen, the rank-4 shared basis is 29.1% smaller than independent while just missing interface quality; top-8 sparse residuals recover quality but reduce savings below the preregistered limit. An oracle-selected residual upper bound does not establish that a learned physics view can discover these entries. The Mirror coordinate is ordinary per-regime PCA coefficients plus sparse values and has no incremental value over that native control.

## Hypothesis result

**FAIL** for Mirror-specific value and for the joint 25% storage / quality gate. Fresh/audit was not opened after exact native aliases and gate failure.

## Frozen protocol / scope

PA186 FNO, PA187 DeepONet and PA18 VeRA. Eight analytically generated 1D variable-coefficient diffusion Green operators (24 interior points); rank-4 PCA basis fit on six, two held out; 256 smooth random forcing vectors per domain; private budgets k={0,4,8,16,32,64} entries per domain. Held-out PCA coordinates and private residual values/support are projected/selected using the known generated Green matrices: this is an oracle mechanism/compression screen, not adaptation from held-out examples. The VeRA-like random-basis row is a simplified shared random matrix control, not a full VeRA training reproduction. Actual NPZ payload bytes include shared operators, codes, all residual indices/values and metadata. Timing is CPU matrix-application only; no optimizer or training updates.

No trained FNO/DeepONet, natural regime, experimental PDE, learned residual selection, discretization transfer or accelerator performance is established.

## H / T / D / C / U

- **H:** Shared operator coordinates and sparse private physics residuals cover common and rare regimes at lower complete bytes than independent models.
- **T:** Seed 73601, 8 variable-diffusion Green operators, N=24, rank 4 fit on 6 regimes, 2 held out; independent, shared mean, PCA, simplified random shared basis, Mirror aliases and sparse residual frontier.
- **D:** **FAIL.** k=8 is the first tested budget meeting both held-out quality limits but saves 20.7%, below the 25% gate. Mirror exactly aliases the native coefficient/private-residual representation.
- **C:** Shared-basis/PCA compression and oracle sparse residual selection explain the observed frontier; no additional Mirror mechanism is needed.
- **U:** Learned operator training, inference-time adaptation from real sensor/forcing observations, natural regime transfer, actual PDE validation and runtime on accelerator.
