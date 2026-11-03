# MA-627 — Periodic function code with phase/frequency modulation

Status: SCREENING; prior art PA132. Dedicated branch: `research/ma-627-periodic-mirror-modulation-20261103`.

## H

A shared periodic decoder with compact amplitude/phase/frequency coordinates can recover held-out 1D signals using fewer bytes than independent sampled curves. Mirror-specific value requires beating standard periodic activation modulation with the same coordinates.

## T

Each seed creates a 4×4 family over harmonic frequency 1–4 and phase 0, π/2, π, 3π/2, with diagonal combinations held out. Amplitude is task-specific and support-inferred. Use 64 support and 512 disjoint audit points; fit by selecting the best of four harmonic frequencies and solving sine/cosine coefficients on support. Compare shared Mirror periodic decoder, native modulated periodic activation (same code), and independent 512-point FP32 sampled curves. Measure actual `.npz` bytes, support-search work, evaluation operations and wall time. Fresh seeds 62711–62713 sealed after a native alias.

## D

Pending.

## C / U

Counter-hypothesis: this is standard periodic activation modulation under different naming. It does not establish performance for general neural implicit representations or learned multi-layer SIREN decoders.
