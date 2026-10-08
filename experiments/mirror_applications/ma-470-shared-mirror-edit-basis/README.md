# MA-470 — Shared Mirror edit basis + MEND coefficient generator

Status: SCREENING (protocol pending freeze)  
Branch: `research/ma-470-shared-edit-basis-20261008`  
Base commit: `94cdea8af94aaf2345870cc7e89c6b08df725f07`  
Prior art: PA86 (MEND)

## H — Hypothesis

A MEND-style editor can produce two reusable coordinates for each of many local edit requests whose updates share two matrix atoms. This may reduce actual edit-bank bytes versus storing a full update per request while preserving edit success and locality. The direct native basis/coefficient control determines whether the parameterization adds anything beyond ordinary low-rank storage.

## T — Frozen protocol

A frozen 24×24 linear base has 20 simultaneous edits. Their key anchors are the first 20 coordinate basis vectors in R^24; all keys are stored and paid in each payload. Sixteen support signals train the shared editor and four are held out. A common nearest-key router with radius 1.8 is applied to every method; key vectors and routing metadata are paid in each actual inference payload. Queries are generated near their own key (Gaussian noise σ=0.18); locality inputs are independent standard-normal vectors. Development seeds are 47001/47002. Fresh seeds 47011–47013 remain sealed until all frozen gates pass. See `PROTOCOL.json` for exact controls, operations, gates and storage rules.

PA86 uses learned transformations from low-rank edit-gradient signals into parameter updates. MA-470 isolates shared two-atom coefficient storage under a common explicit local edit router. It is a synthetic mechanism screen, not factual editing evidence.

## Results

Pending frozen development runs.
