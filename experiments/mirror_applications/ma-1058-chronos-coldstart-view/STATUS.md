# MA-1058 status

Current phase: development complete; fresh series audit protocol is frozen, audit IDs 62–77 remain unread.

Prior art PA326/PA332 is reviewed. The experiment uses official Chronos-T5-Tiny at pinned revision `29d808298f1a62493e7b9a5e08529d0d930fa189` and PEMS-BAY at `c858397fb381d82c3d55ad0a81beb51a9bb9bdaa`. Fit sensors 22–37, dev sensors 38–45, audit sensors 62–77. IDs 46–61 were discarded after an accidental finite-value-only inspection; AM-01 records this and all later HDF reads select columns explicitly.

Development selected 60 support updates for each per-series method. Fit uses 30 updates for the shared rank-2 basis. Dev macro MASE: native Chronos 4.152; pooled LoRA 4.178; one-angle Mirror 3.912; diagonal gain 3.913; full two-coefficient 3.860; independent per-series rank-2 LoRA 3.872; persistence 3.903; seasonal naive 6.683. Mirror and diagonal differ by about 0.04%, within the preregistered 0.1% alias/failure band. The richer direct and independent controls are slightly better; persistence is also marginally better. These are development facts only; fresh audit remains necessary for the final decision.

The first development runner pass is preserved under `runs/dev/invalidated_prepaired/` and is not used for selection. Current selection and model hashes are in `runs/dev/selection.json` and `runs/dev/development.json`.
