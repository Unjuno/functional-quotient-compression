# MA-258 — Expert-bank superposition and Mirror unbinding

Overall registry status: **PROMISING, narrowly scoped**. The canonical experiment is on remote branch `research/ma-258-psp-mirror-expert-bank-20261008` (result commit `82263f49d4d0d5af96f4ee715ac7a6622b60da71`). This branch adds a protocol-distinct flattened-parameter sensitivity rerun; it does not replace the canonical result.

## H

For related expert maps, an expert-addressed orthogonal Mirror view can preserve functions with lower serialized state than independent experts, PSP and simple shared-basis controls; unrelated maps reveal where private state is needed.

## T

**Canonical screen:** eight 16x16 linear experts; Mirror rotates two output channels per expert. Three aligned and three unrelated development worlds selected rank-2 SVD; three fresh worlds were opened after its preregistered gate passed. Controls were independent experts, hard tying, rank-1/2/4/8 SVD, and PA16 packed-sign PSP. Expert IDs were oracle supplied; zero optimizer updates. Actual deterministic NPZ payloads were reloaded.

**Supplemental strict rerun in this branch:** a separately frozen geometry rotates the full flattened 256-parameter matrix, with development seeds 25821/25822. Its cap required total Mirror payload <=0.50x rank-2 SVD; fresh seeds stayed sealed when this gate missed. Existing artifact/cache files were preserved and excluded.

## D

**PROMISING at the canonical output-channel Givens scope.** The canonical fresh aligned runs reported Mirror normalized MSE 8.99e-10–1.07e-9 at 1,514 B, rank-2 SVD MSE about 9e-15 at 3,822 B, and independent experts at 8,440 B. PSP had substantial interference. Canonical fresh unrelated Mirror MSE was 0.848–0.889, similar to hard tying; independent experts remained near zero. No runtime advantage was established. Canonical replay verified 60 payload rows exactly.

**Supplemental rerun: FAIL under its stricter total-byte gate.** Flattened-orbit Mirror reproduced aligned matrices at zero measured MSE using 1,327 B versus SVD 2,592 B (0.512x), narrowly missing the frozen 0.50x cap. Unrelated Mirror max expert normalized MSE was 0.867–0.894. Twenty clean development payloads replayed exactly; fresh remained sealed.

## C

The canonical aligned targets and supplemental aligned targets were generated from the same class of orthogonal view action tested by Mirror. Those are favorable oracle representation screens, not learned expert or MoE results. Rank-2 SVD reconstructs the canonical aligned bank at equal quality but uses more bytes. The supplemental full-vector orbit misses its stricter byte cap.

## U

Natural or trained expert banks, learned routing, nonlinear experts, end-to-end runtime, and near-converged capacity remain untested. The unrelated family shows a strong private-state boundary in both screens.

## Fact / interpretation / hypothesis

- **Fact:** Canonical branch passed its registered gates and fresh replay; supplemental flattened-parameter run missed its own byte cap and had poor unrelated-map fit.
- **Interpretation:** Structured expert orbits can be compactly represented, but the exact interface and actual archive overhead affect the storage frontier; unrelated experts need more state.
- **Hypothesis:** The canonical output-channel view may be useful where trained experts share that structure. Natural-task evidence is absent.
