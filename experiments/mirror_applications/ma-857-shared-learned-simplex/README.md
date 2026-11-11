# MA-857 — shared learned simplex task vertices

This screen tests whether jointly trained MLP endpoints define a useful low-loss simplex across three rotated Digits domains. Training samples barycentric coordinates and trains the interpolated model on examples drawn from the matching domain mixture. Fresh seeds and controls are fixed in `PROTOCOL.json`.

Mirror is encoded as `m = 2a - 1`; decoding back to barycentric weights must match native interpolation. Actual serialized payload bytes include endpoints, codes, and architecture metadata. See `RESULTS.md` after the fresh run.
