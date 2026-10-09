# MA-352 — MIMO boundary-head compression with Mirror

Status: SCREENING. Prior art: PA42 MIMO implicit ensembles.

## H

With a common MIMO trunk evaluated over all member inputs, compressing only aligned member-specific output heads into a shared head plus Mirror phases may preserve member quality/diversity while reducing total trunk+head payload. A generic shared-basis control may erase the Mirror-specific advantage.

## Mirror insertion

> **Mirror insertion:** this experiment adds a member phase `m` to a shared output head after a common hidden trunk, replacing only the per-member head bank while retaining the shared trunk and multi-input/output training.

Compare shared trunk+full MIMO heads, shared trunk+Mirror head views, shared trunk+generic coefficient basis, and a single shared head. Every payload includes the identical trunk, head state, codes and metadata. Evaluate held-out member NLL/ECE/accuracy/disagreement, total bytes, head bytes separately, and one-batch latency including trunk and view reconstruction.

## T

16-input, 64-hidden ReLU trunk, eight binary member tasks, teacher heads on a planted planar rotation orbit. 256 training and 1,024 test examples/member. Train all methods jointly for 1,000 Adam updates using the same multi-member input batch. Development worlds 35221–35222; fresh worlds 35231–35233.

## Gates

**PROMISING:** Mirror NLL within .02 and disagreement within .02 of MIMO; total payload <=.80x MIMO and <=.90x generic basis; single batched inference retained. **FAIL:** total payload gate misses, quality/diversity falls, or simple basis matches within 10% bytes/quality.

## C / U

The random shared trunk is frozen and the teacher heads are intentionally rotation-aligned. This isolates output-head storage but is not a trained deep MIMO reproduction. Natural head diversity and end-to-end training are untested.
