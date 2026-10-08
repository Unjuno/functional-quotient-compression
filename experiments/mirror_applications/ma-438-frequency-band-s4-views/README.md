# MA-438 — frequency-band Views over a shared S4 kernel

Status: SCREENING  
Evidence lane: MECHANISM / STORAGE / QUALITY  
Base commit: `eac3d8e`

## H — falsifiable hypothesis

A shared stable multi-pole S4 kernel plus small per-role Mirror coordinates that shift pole timescales will recover several band-specific sequence filters with less actual serialized state than independent kernels, while outperforming scalar amplitude gates and direct low-rank pole residuals.

## Mirror insertion

> **Mirror insertion:** this experiment adds a role code `m` to the shared diagonal S4 pole spectrum so that different timescale bands can be represented without storing a full pole spectrum per role.

- Native object: stable diagonal S4D kernel with shared input/output factors.
- Mirror: role-specific low-dimensional additive offsets to log-pole decay rates; the discrete convolution kernel is generated from the shared spectrum and `m`.
- Persistent coordinate: two coefficients per role on fixed slow/fast spectral basis vectors.
- Closest controls: scalar output gate, per-role rank-two pole residual, independent pole spectra.

## PA74 delta

PA74 establishes structured SSM kernels and recurrent/convolutional evaluation. MA-437 tested factor rotations of normal-plus-low-rank transitions. MA-438 isolates frequency specialization across timescales in a diagonal S4D kernel; it asks whether the small `m` code beats native pole residuals in actual payload bytes and held-out sequence response.

## Protocol and gates

Synthetic stable four-band sequence filtering; train horizon 64, test horizons 64 and 128. Four roles correspond to slow, mid, fast, and broadband decay targets. Compare shared kernel, scalar gate, shared-kernel-plus-Mirror spectral shifts, equal-rank pole residual, and independent kernels. Development worlds 43800/43801 choose LR in {0.003,0.01}; fresh worlds 43810/43811/43812, seeds 0/1/2. Same 500 updates and batch 64 for every method.

PASS requires fresh horizon-128 NRMSE <=1.10x independent in every world and actual bytes <=60% of independent; every pole stable; no more than 1.25x active MAC proxy. FAIL if any gate fails or the rank-residual control matches quality at lower bytes.

## Storage and compute

Charge serialized state dict bytes for poles, coordinates, gates/residuals, and metadata. Report active MAC proxy, training wall time, and per-sequence convolution evaluation time. This is a synthetic mechanism screen, not an S4 implementation or language-model claim.

## C — strongest counter-hypothesis

Native diagonal pole residuals are already the simplest representation of timescale changes; the Mirror parameter may be a reparameterization that adds codes and compute without lowering total bytes.

## U — unresolved

Natural S4 kernel banks, learned timescale mixtures, FFT kernels, long-range language tasks, and accelerator runtime remain untested.

## Results and decision

**D — PROMISING, narrowly scoped, strict byte gate missed.** Fresh horizon-128 means over three worlds and three seeds: Mirror NRMSE 0.000041 at 2,061B; independent poles 0.000075 at 2,293B; rank-two pole residual 0.000275 at 2,053B. Mirror is more accurate than independent while using 10.1% fewer actual bytes and is substantially more accurate than the nearly equal-byte rank-two residual. It misses the preregistered <=60% payload gate. Active MAC proxy is 10 vs 8 per step (1.25x, at the gate), and measured CPU sequence-evaluation time is comparable. All poles remained stable (maximum <0.762).

**FACT:** quality and stability gates pass in all three fresh worlds; the registered byte threshold does not. Mirror beats independent on both quality and bytes by a modest margin. Its clear matched-size comparison is rank-two residual: Mirror costs 8B more and reduces NRMSE about 6.7x.

**INTERPRETATION:** the chosen spectral code produces a useful timescale specialization point, but most of the physical kernel remains shared and serialized overhead limits compression. This is a bounded synthetic quality/storage Pareto signal, not an S4 serving result or capacity claim.

**H — tested:** low-dimensional role codes shifting a shared diagonal S4 pole spectrum recover band-specific filters with lower bytes than independent kernels.

**T — executed:** 8-pole diagonal recurrence; four role-specific timescales; 500 AdamW updates × batch 64; development worlds 43800/43801 chose LR 0.003 for shared/gate and 0.01 for Mirror/rank2/independent; fresh worlds 43810/43811/43812, three seeds each; horizons 64 and 128; actual torch inference payload bytes and recurrent MAC proxy.

**C — strongest counter-hypothesis:** the teacher is intentionally aligned to log-pole shifts, and the byte gain comes partly from sharing. Natural S4 kernel banks may favor native pole adaptation or require more private coefficients.

**U — unresolved:** natural S4 kernels, FFT implementation, long-range language quality, and accelerator runtime.
