# MA-1030 — SAM2 rare-object memory private fallback

**Decision: FAIL — scoped to the preregistered synthetic SAM2.1 Hiera-tiny CPU test.**

## H — Hypothesis

One causal private-memory slot selected from each frame's object-to-shared-memory residual would preserve a rare appearance-shifted track, remain within 1 percentage point of independent memory, and use at least 25% fewer object-memory bytes than independent memory, without consulting future masks.

## T — Run

- Official SAM2.1 Hiera-tiny source commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`; checkpoint SHA-256 `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`; 256 px CPU path.
- Four prompted objects, six frames per world. Object ID 99 changes appearance beginning at frame 3 during a short occlusion. Dev seed 103000 selected one predeclared residual threshold from `{0.15,0.30,0.50,0.80}`. Frozen threshold 0.15 and development records were committed before any audit worlds were generated.
- Fresh audit seeds 103001, 103002, 103003. Controls: independent, hard-shared mean memory, two-angle Givens, rank-1 additive residual, and Givens with a maximum-one object private memory per frame. Selector observes only current SAM2 memory tensors; no future frame, future mask, or future prediction error is read.
- The full session charge includes actual checkpoint, config, normalized video-frame cache, prompts, masks, pointers, scores, shared bases, coordinates, fallback indices and private tensors. Serialized payloads and shared caches were retained and hash-checked. No training updates; threshold selection is development-only.

## D — FAIL

### Fact

| Method | Mean J&F | Object-memory bytes | Full-session bytes | ID switches by seed | Mean FPS |
|---|---:|---:|---:|---|---:|
| Independent | 0.60567 | 786,432 | 162,992,597 | 0, 0, 4 | 3.472 |
| Hard shared | 0.01105 | 196,608 | 162,402,761 | 0, 0, 0 | 3.474 |
| Givens Mirror | 0.02042 | 196,800 | 162,403,389 | 9, 5, 8 | 3.451 |
| Rank-1 additive | 0.28987 | 393,312 | 162,600,359 | 14, 12, 12 | 3.489 |
| Givens + private fallback | 0.17772 | 393,456 | 162,601,061 | 12, 12, 15 | 3.435 |

The selected fallback used one private tensor on all 18/18 available memory-frame slots across the three audit worlds, and selected object 42 each time; it selected the appearance-shifted object 99 **zero** times. The fallback improves over Givens-only J&F in all audit worlds (by 13.8–17.9 pp), but trails independent by 39.5–47.4 pp and has more ID switches in every world. It fails the registered quality and ID-switch gates.

The fallback compresses object-memory tensors by 50.0% versus independent, but complete-session bytes decrease only 391,536 B (0.240%). Rank-1 additive is a stronger control: it has higher J&F in all three worlds, uses 144 fewer object-memory bytes, is 702 B smaller for the complete session, and has higher measured FPS. Fallback encode/decode time averaged 17.1 ms per video versus 1.9 ms for independent memory.

Per-seed measurements and payload manifests are in `RESULTS_CORE.csv` and `runs/audit_rows.jsonl`.

### Interpretation

A residual from the cross-object mean detects a persistently different object representation, not the *change* that signals a newly rare appearance state. It spent its sole private slot following stable object 42 and therefore never protected shifted object 99. Even with a 50% memory-tensor reduction, common model, cache, and per-object output state leave only a 0.24% complete-session saving. The native rank-1 residual control dominates this fallback variant on quality, bytes, and throughput.

### Counter-hypothesis (C)

The synthetic objects have deliberately similar colors and include occlusion; SAM2's independent memory itself reaches only 0.606 mean J&F. This is a stress test of the online allocation rule, not a claim about real-world object tracking. The fallback failure could depend on this single synthetic scene design.

### Unknown (U)

Natural videos, longer sequences, DAVIS/YouTubeVOS, Hiera-large, GPU throughput, recurrent or temporal-change-aware fallback, more than one private slot, and learned residual codes remain untested. The current detector is not evidence against all private-memory allocation strategies.

## Verification

Development threshold freeze commit: `2f3d44166b663c8c551d552640b2345a15dd444e`, before audit data generation. Saved prediction metrics replay within `1e-12`; all audit input, prediction, cache and session-payload hashes and tensor lengths verify; complete-session byte arithmetic verifies. Four tests pass. Checkpoint remains outside Git with exact hash recorded in `VERIFICATION.json`.
