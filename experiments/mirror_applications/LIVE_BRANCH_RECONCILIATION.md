# Live MA branch evidence reconciliation

Fetched current `origin/research/ma-*` refs on 2026-10-08 and reconciled terminal per-ID claims into the latest 1155-row worker-ready registry. The detailed manifest records each source branch, HEAD, status, and report/verification path. The latest terminal branch record wins only when registry and claim ledger agree and both report and verification artifacts are locatable. Conflicting statuses are disclosed in `alternative_statuses`; every source branch remains unchanged.

MA-325 was separately marked NOT ESTABLISHED from its explicit status report because its independent upper control remained at chance; its verification states that metric replay was not checked. This is not a FAIL verdict for Mirror.

MA-369/371/372 remain paused after the documented consecutive width/depth family failures. The first untested P0 outside that pause is MA-401 (PA63); live remote search found no MA-401 branch.

No fresh data were opened during reconciliation.
