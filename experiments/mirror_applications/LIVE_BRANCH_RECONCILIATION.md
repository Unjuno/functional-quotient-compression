# Live MA branch evidence reconciliation

Fetched current `origin/research/ma-*` refs on 2026-10-08 and reconciled terminal per-ID claims into the latest 1155-row worker-ready registry; the manifest was subsequently extended with the MA-401, MA-403 and MA-405 result branches. A separate newer-named MA-403 branch contained only an UNTESTED registry row and no experiment artifacts; it is recorded as a non-terminal alternative, not an outcome. The detailed manifest records each source branch, HEAD, status, and report/verification path. The latest terminal branch record wins only when registry and claim ledger agree and both report and verification artifacts are locatable. Conflicting statuses are disclosed in `alternative_statuses`; every source branch remains unchanged. This extension adds verified MA-442, MA-444 and MA-451 FAIL branches; their result commits are listed in `LIVE_BRANCH_RECONCILIATION.csv`.

MA-325 was separately marked NOT ESTABLISHED from its explicit status report because its independent upper control remained at chance; its verification states that metric replay was not checked. This is not a FAIL verdict for Mirror.

MA-369/371/372 remain paused after the documented consecutive width/depth family failures. After MA-451 was terminally recorded and remote branches refreshed, MA-452 (PA80/PA81) was the next eligible P0 with no remote experiment branch; MA-446 has a terminal FAIL branch.

No fresh data were opened during reconciliation.
