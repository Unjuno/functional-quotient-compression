# MA-001 status

- Status: PROMISING
- Branch: `research/ma-001-mirror-top1-expert-20261007`
- Base commit: `fa635d0f4b3f98528ac2f731b316e2a664289d3e`
- Protocol freeze: `58c82ae5e43ce135a6dbd48a041cefaf465a2645`
- Development complete: yes; selected LR 0.01 by preregistered mean MSE
- Fresh/audit opened: yes; frozen worlds 10001–10003 complete for both teacher modes
- Fresh replay: all 36 deterministic rows matched exactly
- Results committed: pending final result commit
- Verification: 3 tests passed; serialized round-trip checked for every run
- Registry row: PROMISING

## H / T / D / C / U

- **H:** A four-role top-1 router can recover related nonlinear expert functions by applying small role-specific Givens views to one shared FFN, at lower serialized bytes than untied FFNs; arbitrary role functions should need private weights.
- **T:** Six controls × two teacher modes × three frozen fresh worlds; 1,200 AdamW updates, 64 examples/update, LR 0.01 selected only on development world 10000. See README and PROTOCOL.
- **D:** PROMISING. Aligned quality/storage gate passed 3/3. Mirror payload (7,697B) was 66.9% below dense MoE, but 252B above hard tying; preregistered Mirror-specific byte gate therefore failed. Independent functions were not recovered.
- **C:** The aligned teacher was generated from the same Givens-view family. This is a favorable synthetic construction, and generic shared-basis controls were not byte/compute matched.
- **U:** Natural-language transfer, near-converged fixed-byte capacity, optimized kernels, and broader boundaries remain untested. Eager CPU inference throughput was 0.194x hard tying.

## Blockers

None. Result and tracker commit remain to be written and pushed.

## Decisions / rulings

- MA-001 is narrowly scoped to nonlinear single-layer hard top-1 expert dispatch. It extends the already completed MA-003 linear screen and differs from MA-241's nonlinear cross-depth tying screen.
- Development is a gate: if selected Mirror quality exceeds 1.25x untied full-MoE MSE or Mirror bytes exceed 0.65x full-MoE bytes, do not open fresh worlds.
