# MA-318 — Continual coordinate-first skill acquisition

Status: protocol frozen before development. Dedicated branch: `research/ma-318-continual-coordinate-first-20261008`.

## H — hypothesis

In a sequential 12-skill stream, try a compact coordinate in existing shared state first, then add basis directions only after validation failure. This should identify when shared subspace reuse ends and physical growth becomes necessary while retaining prior skills.

## Mirror insertion

PA33 shows low-dimensional coordinates can adapt a shared model; PA30 motivates separating reusable from task-specific subspaces. Here the initial aligned two-vector family receives a Mirror phase. New families can grow the shared basis after a fixed validation threshold; coordinates after growth use ordinary direct coefficients. The nearest non-Mirror control uses the same evolving basis and growth rule with free coefficients.

## Frozen protocol

See `PROTOCOL.json`. Three four-skill groups: initial phase orbit, second orbit, and unrelated functions. Each task has 128 support, 64 validation and 128 test inputs. Development seeds 31801/31802; fresh 31811/31812/31813. No optimizer updates. Actual payload bytes and task retention are recorded after every new skill.

## Development observations

Both development streams triggered zero growth for the initial four tasks, two basis additions for the second two-dimensional orbit, and four more for unrelated tasks (six growth events total). Direct evolving-basis coefficients retained all skills at max test nMSE <=2.1e-7 and used 2,160B final payload. Mirror retained them at <=9.3e-6 but used 2,654B because phase codes plus post-growth coefficients/metadata exceeded the direct representation. Independent FP16 vectors used 2,250B. The frozen Mirror storage gate is therefore missed in development; fresh evaluation remains locked and will proceed unchanged.
