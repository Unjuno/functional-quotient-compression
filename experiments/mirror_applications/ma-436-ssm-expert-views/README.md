# MA-436 — routed logical SSM experts

Status: SCREENING; PA73/PA74 reviewed; amended protocol revision 2 frozen before canonical development rerun.

## H — Hypothesis
Small routed state-view codes can create several recurrent experts from one physical structured SSM while using fewer actual bytes than full expert copies.

## T — Frozen mechanism screen
Eight experts apply four Givens angles to the state basis around one stable 8-state SSM. Each expert receives 24 sequences of length 128. Compare unconditioned shared SSM, native generated weights, and independent full SSM copies. All routing IDs and reconstruction state are charged. No training occurs; this is a structured synthetic recurrence check, not S4/Mamba or language-model evidence. Fresh seeds 43611–43613 remain sealed.

See `PROTOCOL.json` and `PROTOCOL_AMENDMENT_1.json` for the amended recurrence, data split, seeds, payload rules and gates. The initial unamended screen is retained under `runs/initial_invalid_similarity_gauge/` and is invalid because inputs differed across experts and a full A/B/C similarity transform preserves the input-output transfer function.
