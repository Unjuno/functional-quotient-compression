# MA-1091 — Before/after season Mirror state on geospatial features

Status: **BLOCKED BEFORE PROTOCOL FREEZE; registry remains UNTESTED**  
Evidence lane: STORAGE / RUNTIME / QUALITY (not entered)  
Base commit: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`

## Hypothesis

H (registered): a small factorized pre-observation and post-observation Mirror state over one Prithvi/AnySat backbone can recover useful logical temporal change functions while sharing the physical backbone, across geographic and seasonal shifts.

This candidate could not be preregistered without changing its registered estimand. The available OSCD_MSI pair set has geographic city splits but lacks verified acquisition-season/year metadata needed to separate seasonal variation from true change. No model or imagery values were accessed for this blocker assessment.

## Mirror insertion

> **Mirror insertion:** this experiment was intended to add pre- and post-observation codes `m_pre` and `m_post` to the temporal feature interface of one shared Prithvi encoder, so distinct observation roles and change tasks could be expressed without duplicating the backbone.

- Native method: Prithvi-EO-2.0 multi-temporal encoder plus a binary change head.
- Exact Mirror insertion: factorized `m_pre × m_post` modulation of shared temporal features.
- Strongest simple control: matched-byte FiLM/gate on the same shared temporal features.
- Closest prior art: PA347 AnySat and PA348 Prithvi-EO-2.0.

## Blocker evidence

FACT:
- Runtime inspection found CPU-only PyTorch (`2.6.0+cpu`), no CUDA device, and no `timm`, `rasterio`, `pyarrow`, or `datasets` packages installed.
- Public metadata indicates the candidate OSCD_MSI corpus has 24 city pairs (14 train, 10 test) with binary change masks; the inspected metadata does not establish acquisition-season labels or the seasonal/no-change strata required by the registry row.
- Prithvi-EO-2.0 uses 6 multispectral bands and a 4-frame temporal input in its published config; a two-date OSCD pair would require a non-native frame construction or a documented model adaptation.
- The registered claim requires seasonal and geographic generalization plus change detection. No protocol can currently distinguish seasonal effects from change or specify the correct temporal input without selecting an unregistered adaptation.
- No dataset payload, model weights, labels, or audit values were downloaded or inspected. No experiments were run.

INTERPRETATION:
- Running generic binary change detection on OSCD alone would test a different and narrower hypothesis. Duplicating dates to fit four model frames would also need a frozen justification and input control.

HYPOTHESIS:
- A benchmark with verified acquisition dates/seasons and a documented native two-date Prithvi/AnySat path may make this candidate executable. That possibility has not been tested.

COUNTER-HYPOTHESIS:
- A standard shared encoder plus a simple pairwise change head may explain any downstream benefit; the proposed factorized codes may add no value beyond a matched FiLM/gate.

UNCONFIRMED:
- Feasibility of a GPU run; geographic/seasonal split quality; data licenses and exact dataset schema; frozen Prithvi checkpoint loading; native baseline quality; whether Mirror helps; serialized payload bytes; runtime.

## Decision

NOT ESTABLISHED. This is a reproducibility/data-interface blocker before protocol freeze, not a negative Mirror result. The MA-1091 registry status remains UNTESTED. No fresh/audit data were opened.
