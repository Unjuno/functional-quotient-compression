# MA-981 — shared beam codebook with sector Mirror Views

Status: SCREENING — frozen synthetic wireless simulation
Evidence lane: MECHANISM / STORAGE / RUNTIME
Base commit: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`
Selection: Draw23, uniform over eligible P0/UNTESTED candidates; replay in `source/random_draw.json`.

## H — Hypothesis

A shared phase-only learned beam codebook with compact sector Views can retain near-independent sector spectral efficiency while reducing actual payload, and beat byte-near low-rank phase modulation.

## Mirror insertion

> **Mirror insertion:** this experiment adds a sector-specific phase View to each shared ULA beam codeword, so four logical sector beam sets can share the physical codebook state.

- Array: 8-element half-wavelength ULA, one RF chain, phase-only weights.
- View: per-sector phase vector applied to each shared codeword.
- Native method: PA288 joint neural beam codebook.
- Controls: independent sector codebooks, shared global codebook, scalar phase, rank-2 phase correction, fixed DFT.
- Feedback: 3 bits among 8 beams.

## T — Frozen simulation

Four sectors use three-path Rayleigh channels at 10 dB SNR with angular spread around their sector centers. Development seeds are 98101/98102; fresh seeds are 98103, 98104 and 98105. The training schedule, phase-only constraint, rate objective, byte accounting and fixed gates are in `PROTOCOL.json`. This experiment measures simulation outcomes only; it does not claim RF hardware latency or energy.

## C — Strongest counter-hypothesis

Neural codebooks already provide a learned shared beamspace, and a scalar/rank-2 per-sector phase correction may explain any gain at lower or equal bytes. Sector-specific multipath could also require independent private codebooks.

## U — Boundaries

Results apply only to the frozen narrowband ULA simulator. Real CSI, calibrated RF switching, hardware energy, other arrays, SNRs and mobility are untested. Feedback bits and all stored codes are charged.
