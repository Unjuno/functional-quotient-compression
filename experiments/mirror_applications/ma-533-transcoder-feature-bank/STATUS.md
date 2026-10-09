# MA-533 status

- Status: SCREENING
- Branch: `research/ma-533-transcoder-feature-bank-20261009`
- PA103 reviewed; support-only transcoder training and all controls frozen before implementation/development.
- Protocol hash is recorded in `freeze.json`.
- Development seeds 53301/53302 not run; fresh 53311–53313 locked.

## H / T / D / C / U

- **H:** a task-trained sparse transcoder's decoder atoms can provide a task-aligned shared basis for relation FVs, unlike the pretrained SAE basis that failed MA-526–530.
- **T:** pinned Pythia-70m layer3 MLP; train a 2048-feature top-32 transcoder using only tasks0–11 support prompt tokens; choose shared decoder pool16 on fit FVs; code all task FVs; compare explicit FVs, global transcoder OMP16, and global SAE OMP16.
- **D:** screening; no development data or metrics accessed.
- **C:** the task FVs may not lie in the layer3 MLP output atom span; fitting the transcoder may also be too costly or increase actual payload.
- **U:** transcoder fit stability, FVU, FV reconstruction, causal quality, bytes, compute and replay remain unknown.
