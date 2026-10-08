# MA-419 status: FAIL

## H — falsifiable hypothesis

Direct per-signal amplitude/phase/frequency activation codes would approach a learned PA69-style modulator with fewer actual bytes and improve on generic latent concatenation.

## T — executed

Sixteen zero-mean sinusoidal functions with signal-specific amplitude, frequency, and phase; 64 train and 128 disjoint held-out query coordinates per signal. Compared shared SIREN, direct three-value activation Mirror, latent3 input concatenation, learned latent-to-hidden amplitude/phase/frequency modulation network, and independent SIRENs. 500 AdamW updates × batch 256; 2 development worlds × 3 seeds selected LR 0.01 for all; 3 fresh worlds × 3 seeds. Generator weights and signal codes were serialized and charged.

## D — FAIL

Fresh mean normalized RMSE / actual bytes: shared 0.9795 / 2,525B; Mirror3 0.3476 / 3,345B; latent3 0.2653 / 3,353B; modulation network 0.3271 / 4,433B; independent 0.1526 / 8,349B. Mirror uses 24.5% fewer bytes than the modulation network and has 6.3% higher average error, but the preregistered per-world <=1.1 error gate fails on worlds 41910 (1.19×) and 41911 (1.25×). Latent3 also has 23.7% lower average error than Mirror at nearly identical bytes. No Mirror-specific quality/byte advantage is established.

## C — strongest counter-hypothesis

The direct global amplitude/phase/frequency parameters are too restrictive for the trained shared SIREN; latent concatenation changes hidden features more flexibly. The modulation network has generator overhead but performs close to Mirror after its state is fully charged.

## U — unresolved

Mixed-frequency signals, audio/image fields, more expressive activation codes, longer convergence, and GPU inference throughput remain untested. This is a small periodic-function screen only.

