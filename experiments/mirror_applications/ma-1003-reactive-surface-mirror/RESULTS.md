# MA-1003 screening record

## H — hypothesis

A low-description surface code over a shared readout basis of a frozen common interatomic potential can adapt to reactive surfaces with fewer reference structures and fewer serialized adaptation bytes than independent surface readouts while retaining held-out reactive-region energy and force quality.

## T — what was run

**Fact:** CPU feasibility probe only. Loaded official MACE-MP small checkpoint (32,581,838 B; SHA-256 in protocol) and the public H2/Cu extxyz data (4,220 frames). Three selected 56-atom frames were evaluated in each of four distinct lattice/cell groups. Torch 2.14.1 CPU, MACE 0.3.14, ASE 3.29.0. No optimizer updates; no Mirror/readout adapter or PA301 partial-fine-tune control ran. The 12 observed structures are burned development data. Wall time was measured per three-frame group, but it is a coarse cold/warm mixture.

**Fact:** frozen checkpoint energy MAE was 0.770–0.793 eV/atom across these tiny groups. Mean per-atom force-vector norm error was 0.128–0.225 eV/Å. The full rows are in `RESULTS_CORE.csv`.

**Interpretation:** the checkpoint is executable but materially mismatched to this reactive H2/Cu domain; a valid transfer baseline is needed. These values do not measure generalization, adaptation, transition-state performance, or a storage/quality frontier.

## D — decision

**NOT ESTABLISHED.** This screening did not test the hypothesis. The current extxyz does not retain trajectory/path identifiers or explicit barrier labels, and the exact PA301 fine-tuning condition has not been reproduced. No quality or compression conclusion follows.

## C — strongest counter-hypothesis

The apparent headroom may reflect a poor/mismatched common prior rather than a useful few-shot coordinate. Native fine-tuning or a direct ordinary readout could recover the errors with similar or fewer bytes. Any low-rank `B @ m` correction is likely to alias a standard low-rank readout exactly.

## U — remaining unknowns

- Whether disjoint NOMAD mainfile/path groups can be mapped to curated extxyz frames or fetched as a clean split.
- Whether PA301's frozen-transfer recipe is reproducible with this exact prior and data.
- Whether reactive/transition-state geometries are sufficiently represented and which structures form a verifiable path/barrier.
- Fresh few-shot energy and force errors; model/readout serialized bytes; training examples/updates; inference MAC and wall time.

## Fact / interpretation / hypothesis separation

- **Fact:** provenance, hashes, 12-frame probe, metrics, and missing lineage/labels above.
- **Interpretation:** frozen prior exhibits substantial domain error in the sampled structures.
- **Hypothesis:** shared readout basis plus few-shot code could reduce that error, but has no experimental support yet.
