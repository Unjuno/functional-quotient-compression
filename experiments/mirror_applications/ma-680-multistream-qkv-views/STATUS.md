# MA-680 status

- Status: FAIL (Mirror-specific claim)
- Branch: `research/ma-680-multistream-qkv-views-20261104`
- Protocol frozen before development: yes
- Development: seeds 68001/68002
- Fresh/audit: sealed after exact native alias

## H — Hypothesis

Four causally isolated streams can share QKV projections plus a rank-1 stream view, preserving causal-attention outputs while reducing complete bytes versus independent per-stream QKV. No future-token or cross-stream leakage is allowed; Mirror must beat native shared-basis coefficients.

## T — Treatment and controls

Synthetic one-head causal attention, S=4, D=16, sequence length 12, 32 sequences per stream, two seeds. Compared independent QKV, hard tying, rank-1 shared QKV base plus role codes, and native shared-basis coefficient control. Tests replay from serialized payloads, perturb future tokens, and perturb other streams.

## D — Decision

FAIL for Mirror-specific value. Rank-1 views reproduce all attention outputs exactly with zero future and cross-stream leakage. Payloads were ~6.6KB versus ~11.8KB independent (about 56%). Hard tying degrades output (NRMSE .66/.81). Native role coefficients exactly match function and output with zero leakage; files are 12B larger due to field naming. The view path adds a small one-time coefficient reconstruction MAC cost. Fresh stayed sealed.

## C — Strongest counter-hypothesis

This is ordinary low-rank factorization of stream-specific QKV deltas. Native shared-basis role codes explain the full measured byte gain.

## U — Unconfirmed

No trained multi-stream LLM, learned stream routing, cross-stream communication, language metric, or accelerator latency is tested. Causal isolation here is enforced structurally by independent stream attention.

## Evidence labels

- Fact: exact serialized payload bytes, replayed NRMSE, MAC proxy, CPU wall time, and leakage are in `RESULTS_CORE.csv` and `source/development.json`.
- Interpretation: aligned stream-specific QKV deltas can be stored compactly without leakage in this isolated synthetic block.
- Hypothesis: Mirror-specific multi-stream value, if any, must exceed native role-conditioned QKV factorization in a trained cross-stream setting.
