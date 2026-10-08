# MA-457 — Continual path reuse before module birth

Status: SCREENING  
Evidence lane: MECHANISM/STORAGE/CONTINUAL_TRANSFER  
Branch: `research/ma-457-continual-path-reuse-20261008`  
Base commit: `2264c54`  
Prior art: PA80 (PathNet)

## H — Hypothesis

For a task stream with six functions sharing a rank-one update direction and two unrelated outliers, reusing a shared module with a small task coordinate before birthing a full module should preserve task quality with fewer module births and lower actual bytes than a PathNet-style reuse-then-birth policy. A direct rank-one update control measures whether the effect is simply ordinary low-rank adaptation.

## T — Frozen setup

Eight synthetic 4D linear tasks arrive sequentially. Tasks 0–5 share a seeded base matrix plus scalar multiples of a shared rank-one basis; tasks 6–7 are independent matrices. Each task has 256 support and 256 query examples with σ=0.01 noise. The Mirror basis is learned on the first four aligned tasks. New task codes receive 300 updates; the system births a private full module if support RMSE remains above 0.05. Development seeds are 45701 and 45702; fresh seeds 45711–45713 remain sealed. See `PROTOCOL.json` for all frozen gates and counts.

## Mirror insertion

> **Mirror insertion:** store one shared module and shared rank-one basis, and give each task a scalar coordinate `m` before deciding whether a full private module is needed.

PA80 establishes PathNet's task-specific pathway selection, module reuse and freezing. This experiment measures whether task Views delay module birth while retaining old tasks. The simplest control is exactly the same function written as a native rank-one/LoRA coefficient.

## Results

Pending frozen development runs. No conclusion is recorded before metric replay and actual payload verification.
