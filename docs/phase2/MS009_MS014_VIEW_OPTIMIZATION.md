# MS009–MS014 View optimization and shared-world frontier

Date: 2026-10-07 JST
Evidence boundary: small synthetic MLP sensor worlds only. No Transformer, real-sensor, LLM, or general compression claim.

## Main findings

1. Sensor identity should not be fed directly into the shared world core after calibration. MS008 showed worse cross-sensor changed-law transfer in 8/8 fresh worlds when sensor ID entered the core.
2. Residual Views are useful mainly when affine calibration leaves meaningful nonlinear sensor mismatch. Always-on Views are unnecessary in well-calibrated affine worlds.
3. MS009: learned stretch/combined Mirror residuals beat comparable learned gain/FiLM/low-rank controls on the prediction+transfer balance in 10 fresh worlds.
4. MS010: uniform L2 regularization is not a good automatic pruning rule; it also shrinks useful high-mismatch Views.
5. MS012: calibration-residual hard pruning correctly separated low/high mismatch trained sensors in 10/10 fresh worlds. Training all Views first, then pruning and consolidating beat pruning from the start on seen error in 7/10 and transfer in 8/10, but remained slightly worse than keeping all Views for local seen quality.
6. MS013: reducing only the low-residual sensor View rank gives a smooth quality/storage tradeoff. Low rank 2 retained strong improvement over shared-only, while full rank 8 had the best local seen quality.
7. MS014 actual-byte frontier: three independent width-48 sensor models used 33,819 B and had the best seen prediction. Shared+View width48 used 12,079 B (2.80x smaller) but median seen NMSE was 1.897x the independent control. Widening shared+View to width80 used 29,489 B (87.2% of independent storage) but still had 1.646x median seen NMSE. Quality-equivalent compression was NOT reached.
8. Functional sharing was very strong: after changed-law adaptation using labels from sensor0 only, independent receiver models remained at median NMSE 0.4665, while shared/View receivers were around 0.0083-0.0090 without receiver updates.

## MS012 learn-many-then-prune

- always rank8 / shared: median seen ratio 0.7157, transfer ratio 0.9733, 11,783 B.
- learn -> prune -> consolidate / shared: seen 0.7303, transfer 0.9758, new-sensor ratio 0.9419, 11,717 B.
- learn -> prune beat prune-from-start: seen 7/10, transfer 8/10.

## MS013 rank allocation

High-residual trained sensor kept rank8; only the low-residual sensor was reduced.

| low-sensor rank | seen/shared | transfer/shared | new-sensor/shared | bytes |
|---:|---:|---:|---:|---:|
| 0 | 0.7046 | 0.9768 | 0.8769 | 11,998 |
| 2 | 0.7022 | 0.9795 | 0.9056 | 12,017 |
| 8 | 0.6883 | 0.9917 | 0.9303 | 12,079 |

## MS014 actual serialized-byte frontier

Fresh worlds: 54501-54510. Core training uses 2500 optimizer updates. Shared models consume 32 examples per trained sensor per update; independent models also consume 32 examples/sensor/update. View methods additionally use 400 code-only updates with the core frozen.

| method | bytes | median seen NMSE | seen / independent | median changed-law receiver NMSE |
|---|---:|---:|---:|---:|
| independent w48 x3 | 33,819 | 0.000446008 | 1.000 | 0.466546 |
| shared w48 | 11,482 | 0.00104843 | 2.416 | 0.00873867 |
| View w48 | 12,079 | 0.000821837 | 1.897 | 0.00900736 |
| View w64 | 19,761 | 0.000779898 | 1.729 | 0.00864546 |
| View w80 | 29,489 | 0.000738220 | 1.646 | 0.00832447 |

Decision:
- Functional cross-sensor law sharing: PASS in this synthetic family.
- Large storage reduction at equal local prediction quality: FAIL / NOT YET.
- Learn-many-then-prune curriculum: PROMISING but modest.
- Calibration-residual-driven rank allocation: PROMISING; binary pruning is too aggressive.

Next: improve the View residual family / factorized sensor x role representation rather than simply widening the shared core. Preserve no-sensor-ID shared dynamics and measure actual serialized bytes, local quality, and source-only law transfer separately.
