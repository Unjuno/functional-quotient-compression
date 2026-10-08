# MA-981 — shared beam codebook with sector Mirror Views

Status: FAIL — quality passed; payload and simple-control gates failed
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

## Result and worker report

**Fact.** In fresh worlds 98103–98105, Mirror averaged 2.83295, 2.83286 and 2.83673 bit/s/Hz; independent sector codebooks averaged 2.86850, 2.86910 and 2.87194. The quality margin passed (≤0.20 loss), but Mirror's 2,360 B payload was only 14.7% smaller than independent codebooks at 2,767 B, failing the preregistered 25% reduction. Rank-2 correction was 2,557 B, outside the 5% matching window, and Mirror did not beat the scalar phase control: scalar averaged 2.83250, 2.83439 and 2.83654 at 2,232 B. All 30 payloads and rate metrics replayed exactly; no audit data exists for this simulator.

Average measured training time was 0.254 s for Mirror, 0.228 s for independent, 0.248 s for scalar and 0.275 s for rank-2. Measured throughput was 7.72M, 7.81M, 8.28M and 7.88M examples/s respectively. The phase-materialization compute proxy is 64 extra operations for Mirror/scalar and 80 for rank-2 per sector codebook construction; beam scoring is common across methods. A post-run accounting review corrected this proxy only; trained weights and quality metrics did not change.

**Interpretation.** Sector Views recovered nearly the quality of independent codebooks, but the simple scalar control used fewer bytes with comparable rate. Most observed value is codebook sharing already represented by native controls, not a Mirror-specific gain. The simulation provides no hardware switching or energy evidence.

**Hypothesis.** More correlated sector distributions, larger arrays, or a functionally shared beam basis may improve the storage frontier, but this single geometry and channel model do not establish it.

**H** Sector-specific phase Views over a shared codebook would stay within 0.20 bit/s/Hz of independent sectors, cut their payload by 25%, and beat byte-near rank-2 modulation by 0.10. **T** Two development and three fresh 8-element ULA simulation worlds, 500 updates for learned methods, 3-bit selection from 8 phase-only beams, native and independent controls, actual serialization and replay. **D: FAIL.** **C** A scalar per-sector phase and native learned codebook explain the result; independent private beam state may still be required for the remaining quality difference. **U** Real CSI, hardware, RF switching delay, energy, mobility and other SNRs remain untested.
