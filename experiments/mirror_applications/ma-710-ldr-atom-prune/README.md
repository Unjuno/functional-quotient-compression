# MA-710 — Learn-many LDR atoms then Mirror-compose and prune

Status: SCREENING  
Evidence lane: MECHANISM/STORAGE/COMPUTE  
Base commit: `54992634`

## Hypothesis

H: Fit task codes over an overcomplete bank of Toeplitz/LDR atoms, then physically prune atoms that no task uses, preserving held-out functions at a lower complete payload than independent Toeplitz kernels. Random pruning and small-from-start controls test whether post-fit selection provides value.

## Mirror insertion

> **Mirror insertion:** logical task maps are sparse signed combinations of shared LDR atoms; after task assignments settle, unused physical atoms are removed and all indices remapped.

- Prior art: PA174 learned compressed LDR transforms; PA78 sparse explicit conditioning.
- Controls: independent dense and Toeplitz fits; full atom bank; ordinary sparse coding on identical atoms; post-fit pruning; equal-size random prune; small-from-start.
- Exact Mirror-specific test: whether the selection/address formulation beats standard sparse coding or pruning at equal bytes. Exact alias counts against attribution.

## Protocol

The source of atoms, seed groups, support/audit sizes, sparsity, thresholds, byte accounting, and gates are frozen in [PROTOCOL.json](PROTOCOL.json). Complete model payload bytes are measured after serialization. Task support vectors determine codes; audit vectors remain held out.

## Results

Pending.

## H/T/D/C/U

- **H:** Post-fit physical pruning can retain function quality at lower storage than independent kernels and equal-size random/small-start controls.
- **T:** D=32, 24 tasks, 32 atoms, two atoms/task, 128 support and audit vectors, three audit seeds.
- **D:** Pending.
- **C:** This is ordinary sparse coding, dictionary learning, and physical pruning; likely exact native controls.
- **U:** Natural task quality, learned deep LDR generators, and optimized hardware runtime.
