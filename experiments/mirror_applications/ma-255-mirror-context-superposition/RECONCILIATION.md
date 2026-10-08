# MA-255 protocol reconciliation

## Fact

The latest worker-ready registry identified MA-255 as UNTESTED. Two remote branches nevertheless contain completed reports under the same stable MA ID:

1. `research/ma-255-mirror-context-superposition-20261008` (`ceed174e1214d9e92cc51b0ea58489e9b5100a45`) reports a post-fit representation screen with a rank-2 task-code control, four fresh seeds, and a 734-byte Mirror representation. Its metric replay and serialization checks passed; the scoped aligned task case met its preregistered gate.
2. `research/ma-255-mirror-context-superposition-replication-20261008` (`3dab44af5d4746257664ad8057ad47a358179e06`) reports a separate 1,200-update training screen. Its development gate failed and fresh data stayed sealed. The result verification commit is `0a53c9c5a4902c3bd44295c6bcf24098c2f3a97a`.

The protocols are materially different: post-fit decomposition with zero updates versus learned task models trained for 1,200 updates; they also use different payload formats and decision gates. They are not repeated measurements of the same frozen protocol.

## Interpretation and disposition

MA-255 is marked **PROMISING** only for the narrow post-fit aligned representation screen. The fixed-update learning screen remains a valid negative result and is retained in `protocol_variants/fixed_update_screen/`. The positive result does not establish learning efficiency, convergence, natural-task quality, broad Parameter Superposition superiority, or replication of the failed training screen.

The registry now points to the reconciled report and verification in this branch. Future work requiring trained task models must use a new MA ID or an explicit amendment with a protocol that resolves this difference before fresh evaluation.

## Hypothesis and limits

- **H:** a structured task View can encode an aligned family of task maps with useful quality at fewer actual serialized bytes than the implemented fixed-context PSP and rank-2 task-code controls.
- **T:** preserved post-fit screen; development seeds 11/23, fresh seeds 101/211/307/401, zero optimizer updates. The separate 1,200-update development-only FAIL is recorded as a different protocol variant.
- **D:** PROMISING for aligned post-fit representation only; no adoption claim.
- **C:** the teacher exactly matches the one-angle rotation chart, and stronger task-code bases or fuller PSP context families may close the gap.
- **U:** trained-model recovery under a common protocol, stronger PSP reproduction, matched-byte optimization, natural-language quality, near-convergence and scaled runtime.
