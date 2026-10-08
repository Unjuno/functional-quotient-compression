# MA-318 — Continual coordinate-first skill acquisition

Status: FAIL for the Mirror storage/compute gate. Dedicated branch: `research/ma-318-continual-coordinate-first-20261008`.

## H — hypothesis

In a sequential 12-skill stream, try a compact coordinate in existing shared state first, then add basis directions only after validation failure. This should identify when shared subspace reuse ends and physical growth becomes necessary while retaining prior skills.

## Mirror insertion

PA33 shows low-dimensional coordinates can adapt a shared model; PA30 motivates separating reusable from task-specific subspaces. Here the initial aligned two-vector family receives a Mirror phase. New families can grow the shared basis after a fixed validation threshold; coordinates after growth use ordinary direct coefficients. The nearest non-Mirror control uses the same evolving basis and growth rule with free coefficients.

## Frozen protocol

See `PROTOCOL.json`. Three four-skill groups: initial phase orbit, second orbit, and unrelated functions. Each task has 128 support, 64 validation and 128 test inputs. Development seeds 31801/31802; fresh 31811/31812/31813. No optimizer updates. Actual payload bytes and task retention are recorded after every new skill.

## Development observations

Both development streams triggered zero growth for the initial four tasks, two basis additions for the second two-dimensional orbit, and four more for unrelated tasks (six growth events total). Direct evolving-basis coefficients retained all skills at max test nMSE <=2.1e-7 and used 2,160B final payload. Mirror retained them at <=9.3e-6 but used 2,654B because phase codes plus post-growth coefficients/metadata exceeded the direct representation. Independent FP16 vectors used 2,250B. The frozen Mirror storage gate is therefore missed in development; fresh evaluation remains locked and will proceed unchanged.

## H/T/D/C/U report

**H — Hypothesis.** Trying a compact Mirror coordinate first and growing the shared physical basis only on validation failure would preserve all prior tasks, trigger growth only for new subspaces/off-orbit tasks, and reduce final bytes versus ordinary direct coefficients.

**T — Trial.** Twelve 64D linear skills in sequence: four tasks on one planted two-vector orbit, four on a second orbit, and four unrelated vectors. Each task had 128 support, 64 validation and 128 test examples. Methods were hard tying, direct least-squares coefficients with validation-triggered rank-one basis growth, Mirror phase on the initial pair followed by direct coefficients after growth, and independent FP16 vectors. Development seeds 31801/31802; fresh seeds 31811/31812/31813; zero optimizer updates. Every checkpoint payload was serialized and reloaded before evaluating all prior tasks.

**D — FAIL for Mirror storage and compute.** Across all three fresh streams, basis growth occurred six times: twice when the second orbit arrived, then once per unrelated skill. Growth followed validation nMSE 0.44–1.21; no growth occurred for the initial orbit. Direct evolving coefficients retained every task at max test nMSE 1.62e-7–2.33e-7 using 2,160B final payload. Mirror also retained every task (max 4.80e-6–9.69e-6) and had the same six growth events, but used 2,654B: 22.9% more than direct and 18.0% more than independent FP16 vectors (2,250B). Mirror fit proxy was 201,883,648 vs 622,592 for direct (~324x). The quality/growth mechanism worked, but the frozen storage gate failed in 3/3.

**C — Strongest counter-hypothesis.** Basis growth already provides reusable directions and direct coefficients encode them more compactly. The Mirror phase is redundant after the task has a two-coefficient representation; phase arrays, mode IDs and padded post-growth codes add more bytes than they remove, while grid search costs much more than least squares.

**U — Unverified.** This is a planted post-fit linear skill stream with oracle subspace structure, not online neural-network training or a natural continual-learning benchmark. The coordinate fit is not learned jointly with a backbone, and the 12 tasks do not establish independent capacity.

### Evidence categories

- **Fact:** 144 serialized checkpoints were byte/hash checked and exactly replayed across three fresh seeds; all prior-task metrics reproduced with max difference 0. Four tests pass.
- **Interpretation:** Validation-triggered basis growth identifies when physical sharing stops being sufficient. The tested Mirror coordinate did not compress the nearest ordinary coefficient representation and incurred much higher fit compute.
- **Hypothesis:** A coordinate family whose code dimension remains smaller after each growth event, rather than switching to padded direct coefficients, may change this frontier; that would require a separate protocol and control.
