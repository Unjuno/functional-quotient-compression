# MA-545 status

**FAIL — Mirror-specific attribution.** Frozen protocol completed on development seeds 54501–54502 and fresh seeds 54511–54513. Both development gates passed before fresh access; `runs/dev/DEV_GATE.json` records the gate.

The context router selected the correct task for every query and routed top-1 matched the oracle FV. FV vectors improved gold-answer likelihood on fresh splits but decreased top-1 choice accuracy. Same-router output-bias controls reproduced the FV scores within the preregistered numerical tolerance on every seed. The useful activation intervention therefore aliases a standard native bias mechanism.

All five raw metrics, payloads and split manifests are retained under `runs/`. The 16-MLP reference is storage-only and has no trained quality result. Full per-method latency is not isolated; total evaluation and extraction wall time are recorded.
