# MA-1058 status

Decision: **FAIL (scoped)**. Fresh audit completed on PEMS-BAY held-out series IDs 62–77 with frozen Chronos-T5-Tiny. Mirror misses its registered quality and Mirror-specific gates: diagonal control is better by 0.47%/0.31%; independent per-series rank-2 LoRA improves MASE by 18.8–20.0%. Mirror adaptation state is 35,310 B (6.34% of independent state), but full payload savings are only 1.53% due to the shared checkpoint.

Verification replays all 16 seed×method metric rows and all 32 serialized payload byte totals. Re-drawing serialized forecasts is exact for 11/16 first queries; five stochastic model outputs on seed 105804 differ despite being finite. Saved sample arrays remain the audit metric source; see RESULTS.md and VERIFICATION.json.

Scope: one PEMS-BAY network and calendar range, two sampling draws (not independent worlds), CPU float32. No capacity claim. The result shows an accuracy/state-size boundary for this cold-start series adaptation setting only.
