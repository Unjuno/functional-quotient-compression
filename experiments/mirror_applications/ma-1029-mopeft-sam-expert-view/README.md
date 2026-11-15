# MA-1029 — SAM feature adapter-bank Mirror views

Status: **FAIL**, scoped synthetic feature-space screen. The Mirror view reduces adaptation payload relative to independent adapters, but hard sharing is 188 B smaller at nearly identical quality, and FiLM is better across all three worlds for 380 additional bytes. Full SAM bundle savings are negligible.

This is not an official MoPEFT or MESS reproduction. The tested modules attach to the final frozen SAM ViT-B image embedding for CPU feasibility; they do not reproduce per-layer MoPEFT. All generated examples are synthetic shapes with box prompts.

- Protocol and resource amendment: `PROTOCOL.json`
- Hypothesis, controls, findings and limits: `RESULTS.md`
- Per-domain metrics: `RESULTS_CORE.csv`
- Audit manifests, predictions, payloads and hashes: `runs/`
- Verification: `VERIFICATION.json`, `source/verify_result.py`

Development settings were frozen and committed before audit examples were generated. The audit has three fresh synthetic worlds. Do not interpret this screen as evidence on natural images, MESS, or general SAM adaptation.
