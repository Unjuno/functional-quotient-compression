# MA-701 — Householder Mirror task Views

Status: FAIL  
Evidence lane: MECHANISM  
Base commit: `a6c00329`

## Hypothesis

H: Short per-task Householder products can express multiple orthogonal task Views over one shared linear operator at less actual state than independent operators, while reflection count trades state against matrix-free compute.

## Mirror insertion

> **Mirror insertion:** adds per-task reflection-vector addresses around one shared operator; the View is applied matrix-free at inference.

- Native baseline: independent task matrices.
- Existing PEFT controls: dense OFT matrices and parameter-matched block-orthogonal transformations.
- Cheapest exact equivalent: native Householder parameterization, which is explicitly an attribution control.

## Prior-art delta

PA171 establishes Householder orthogonal parameterization; PA20/PA21/PA172 establish OFT/BOFT/GSOFT precedents. This experiment tests serialized task-view storage and matrix-free compute against those families without claiming orthogonality itself as new.

## Gates

Frozen dimensions, seeds, output gates, and reflection counts are in [PROTOCOL.json](PROTOCOL.json). All transforms, bases, and metadata are charged in payloads.

## Results and decision

**FACT:** Across three audit seeds, Householder views matched independently materialized functions within 1.74e-7. At r=1/4/8, the shared W plus reflection codes used 14.33%/19.01%/25.25% of independent matrix bytes and 1.031x/1.125x/1.25x matrix-vector FLOP proxy. Dense shared-W+OFT matrices used 295,696 B. The corrected parameter-matched block-orthogonal control used 66,330 B; its output NRMSE ranged .287–.701 across reflection counts. Native Householder and candidate payload hashes/functions were identical.

**INTERPRETATION:** Short reflection codes provide a useful storage frontier against full independent matrices, with moderate matrix-free compute overhead. At equal payload around r=8, the tested block-orthogonal approximation loses substantial output quality. Native Householder is already a known parameterization and exactly aliases the candidate, so no Mirror-specific value is established.

**HYPOTHESIS:** Reflection-vector coordinates compactly represent the deliberately aligned transform family; they do not add function capacity beyond ordinary Householder/OFT parameterization.

**BOUNDARY:** Synthetic linear maps generated from Householder products; no trained task, downstream quality, or GPU runtime claim. Initial dense-zero block accounting is preserved and excluded; corrected compact block payload is authoritative.
