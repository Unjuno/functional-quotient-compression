# MA-468 — Polytropon shared/private skill bank with Mirror skill views

Status: SCREENING  
Evidence lane: MECHANISM/STORAGE/QUALITY/COMPUTE  
Branch: `research/ma-468-polytropon-mirror-skill-views-20261008`  
Base commit: `595a429`  
Prior art: PA85 (Polytropon)

## H — Hypothesis

Six aligned logical skills can share three physical matrices through a compact View basis, while four unrelated functions use private skills. A sparse task-skill allocation table composes the skills. The screen compares quality, actual bytes, module count, compute and exact native low-rank controls against an explicit Polytropon-style skill bank.

## T — Frozen setup

The synthetic task uses eight dimensional inputs. Tasks 0–7 each compose two of six aligned logical skills; tasks 8–11 each select one unrelated private skill. The six aligned teachers are formed from three physical matrices plus a shared rank-one update direction. Development seeds are 46801 and 46802; fresh seeds 46811–46813 remain sealed. Each task has 128 support and 256 query examples, and each model receives 4,000 Adam updates × 32 examples. The task-skill allocation table is fixed and serialized to isolate bank storage from router learning.

Controls are an explicit ten-skill bank, three physical skills without views plus four private skills, the same shared-basis view model under a native low-rank label, and an independent full matrix per task. All bases, private skills and allocation state are charged in actual `.npz` payload bytes.

## Prior-art delta

PA85 establishes discrete task-skill allocation over reusable modules. This screen tests whether each shared physical skill can provide two logical functions via a small View before storing another full skill, while retaining private skill state for unrelated functions.

## Results

Pending frozen development runs.
