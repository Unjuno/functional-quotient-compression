# MA-963 execution blocker

## H — Hypothesis

A context-generated low-description code may compress TACOS synapse-local metaplastic state while retaining task-agnostic continual-learning performance. This would concern resumable online-learning state separately from inference weights.

## T — Feasibility review

**Fact.** The PA281 arXiv v1 source bundle was downloaded and hashed (`13e74d2e540c432cbc396ba74922f6af8a571503f87ee42bd458507cca51f7da`). The paper gives the plasticity multiplier `f(m,w)=exp(-|m*w|)`, reference-weight tracking, and heterosynaptic drift at postsynaptic spikes. The supplementary algorithm says to update `m` by `m+Δm`; it lists `Δm=.04/.004` and neural-trace thresholds `m_th=6/5/2`, but does not make the exact per-synapse increment condition/indexing executable or specify enough simulation timestep/sample-duration detail to reconstruct the benchmark faithfully.

**Fact.** Bounded GitHub repository API searches for the exact title, arXiv ID `2409.00021`, and TACOS/SNN neuromodulation returned zero repositories. The arXiv source itself contains no implementation URL. CPU PyTorch is available, but `snntorch` and `spikingjelly` are not installed.

## D — Decision

**NOT ESTABLISHED; registry remains UNTESTED.** The mandatory native TACOS control cannot be reproduced faithfully from the evidence located. This is an algorithm/source blocker, not a hardware blocker. No arbitrary metaplastic SNN was substituted, and no metrics were generated.

## C — Strongest counter-hypothesis

The missing m-update rule may be implicit in an author's code repository that is not indexed by exact repository searches, or may be inferable from the full paper's omitted implementation details. Without the exact update condition, any result could measure a different algorithm.

## U — What would unblock

An official TACOS implementation or complete specification of per-synapse `m` updates, reference-weight and heterosynaptic dynamics, simulation timing, and benchmark streams. Then compare native m/W_ref state, quantization, neuron broadcast, ordinary low-rank compression, and a context-generated Mirror code under identical task-agnostic streams. Report inference bytes separately from resumable learning-state bytes; `m`/`W_ref` savings alone do not compress inference weights.

## Provenance

Paper: `https://arxiv.org/abs/2409.00021v1` (PA281). Paper source bundle SHA256 is in `PROTOCOL.json`. Bounded repository search queries were `TACOS spiking continual learning neuromodulation`, `2409.00021`, and exact paper title; all returned zero repositories.
