# MA-651 — TTT hidden-model Mirror parameterization

Status: SCREENING. This synthetic TTT-like test compares writable hidden-state parameterizations across sequence streams.

Sixteen streams each adapt a linear predictor online for 128 tokens. Their target updates use one of four rank-4 output subspaces. Compare full TTT-Linear, one shared rank-4 basis, private rank-4 state, a shared basis dictionary with stream codes, and native mode IDs. All bases, codes, initialization and recurrent states are charged in actual serialized payloads.

This isolates online-state capacity and does not reproduce the paper's self-supervised TTT-Linear/MLP setup. See `PROTOCOL.json`.
