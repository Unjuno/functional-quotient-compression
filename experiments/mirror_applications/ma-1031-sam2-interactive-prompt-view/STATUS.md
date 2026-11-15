# MA-1031 status

- Status: **COMPLETE — FAIL**, scoped synthetic SAM2.1 Hiera-tiny CPU test
- Branch: `research/ma-1031-sam2-interactive-prompt-view-20261115`
- Protocol commit: `4dc51fe9385685a2a934b1fccd3834de2e510f88`
- Pre-audit runner correction: `a6baa40a580b1350a0300f2c183fd0854cd4f924`, followed by in-memory state hashing at `d4cfb577d6e29a0337a4f58bcea0987d2434f457`
- Development freeze before audit generation: `3c6fa6cd3e9cb2e503ae83a62f1e76cc9bde9fd5`
- Audit seeds 103101–103103: complete

**H:** A shared correction-pair radius plus per-frame midpoint/angle would reproduce direct-click quality with at least 25% lower click-history bytes.

**T:** Four-object six-frame videos; exact first-frame masks; one positive and one negative correction click per object/frame from current errors; direct, 8px grid, Mirror polar, no-correction and full-mask controls.

**D:** FAIL. Mirror prompt history is 503 B vs 544 B direct (7.5% smaller); J&F is 0.77332 vs 0.83345 direct. The 8px grid is 318 B and scores 0.83840. Mirror has the same 40 clicks as direct and saves only 43 B of full-session state.

**C:** SAM2 is tolerant of coarse click coordinates on these simple synthetic scenes; this can favor the 8px grid.

**U:** Natural videos, longer interaction sequences, other click policies, higher resolution and GPU remain untested. This method compresses click history; it does not reduce human interactions.

Four tests pass; saved prediction metrics and all retained artifact hashes verify. See `RESULTS.md` and `VERIFICATION.json`.
