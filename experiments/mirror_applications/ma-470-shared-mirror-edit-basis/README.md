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

### Facts

| Seed | Method | Edit RMSE | Locality RMSE | Wrong route rate | Actual payload | Generation MAC/edit |
|---:|---|---:|---:|---:|---:|---:|
| 47001 | MEND full update | 0.01106 | 0.00000 | 0.00195 | 202,663 B | 36,864 |
| 47001 | Mirror basis code | 0.01025 | 0.00000 | 0.00195 | 85,832 B | 19,648 |
| 47001 | Native low-rank | 0.01025 | 0.00000 | 0.00195 | 85,832 B | 19,648 |
| 47001 | Independent fit | 0.24811 | 0.00000 | 0.00195 | 51,736 B | 576 |
| 47001 | ROME | 0.03203 | 0.00000 | 0.00195 | 51,717 B | 576 |
| 47001 | No edit | 0.04458 | 0.00000 | 0.00195 | 5,376 B | 576 |
| 47002 | MEND full update | 0.01635 | 0.00000 | 0.00000 | 202,663 B | 36,864 |
| 47002 | Mirror basis code | 0.02057 | 0.00000 | 0.00000 | 85,832 B | 19,648 |
| 47002 | Native low-rank | 0.02057 | 0.00000 | 0.00000 | 85,832 B | 19,648 |
| 47002 | Independent fit | 0.63031 | 0.00000 | 0.00000 | 51,736 B | 576 |
| 47002 | ROME | 0.05640 | 0.00000 | 0.00000 | 51,717 B | 576 |
| 47002 | No edit | 0.08606 | 0.00000 | 0.00000 | 5,376 B | 576 |

All methods had zero locality routes over 512 unrelated samples in each world. Mirror's serialized payload was 57.7% smaller and its edit-generation MAC proxy 46.7% lower than MEND. Mirror and direct native low-rank have byte-identical payloads and identical per-edit outputs in both seeds. The independent fit payload is smaller still but its heldout edit RMSE fails the frozen <=0.025 validity bound. Sub-millisecond generation timings vary and are retained in `RESULTS_CORE.csv`; they are not treated as a stable wall-clock finding.

### T — Execution

Two frozen development worlds; 20 simultaneous keyed edits, 16 editor-training requests and four heldout; 2,200 Adam updates per learned editor; MEND/native low-rank/ROME/independent-fit/no-edit controls; actual NPZ bytes and serialized metric replay. Fresh worlds 47011–47013 were never accessed.

### D — Decision

**NOT ESTABLISHED.** The independent upper's heldout edit RMSE (0.24811/0.63031) exceeds the frozen validity bound despite passing locality and route checks. Thus the storage/quality frontier is unverified. The Mirror code exactly aliases ordinary native low-rank coding, so no Mirror-specific benefit is supported even though its fixed-update signal is smaller than MEND.

### C — Strongest counter-hypothesis

Only 24 support examples identify an unrestricted 24×24 update, making the per-edit least-squares upper underdetermined/noise-sensitive. The editor receives supervised synthetic target updates for its 16 training edits, while each independent fit estimates a full matrix from just 24 inputs. That control mismatch prevents interpreting its large error as an unrestricted achievable frontier.

### U — Unconfirmed

Near-converged independent per-edit quality, fresh-world replication, factual editing, and whether a larger support set or constrained independent low-rank fit yields a valid storage/quality frontier remain unconfirmed. Such a task/control change requires a numbered amendment or new MA ID; fresh data was not opened.

