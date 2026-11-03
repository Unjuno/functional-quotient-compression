# MA-626 status

- Status: FAIL for Mirror-specific attribution; strong scoped compression result vs independent grids
- Branch: `research/ma-626-deepsdf-mirror-autodecoder-20261103`
- Base commit: `70d55f30`
- Protocol frozen before development: yes
- Development complete: yes (62601, 62602)
- Fresh/audit opened: no; native exact alias gate failed
- Results committed: pending
- Verification committed: pending

## H / T / D / C / U

**H:** A shared canonical decoder with compact per-shape Mirror codes reproduces held-out functions at fewer actual bytes than independent payloads; Mirror-specific value must beat direct native shape conditioning.

**T:** Two development seeds; 16 translated/scaled 2D circles each, four held out; 64 support and 256 audit points per shape. Optimized `(cx, cy, r)` from support with least squares. Compared canonical shared decoder with Mirror coordinates, direct native shape conditioning using the same codes, and independent 64×64 FP32 SDF grids. Complete `.npz` inference payloads were measured.

**D — FAIL for Mirror-specific claim.** Mirror and direct native conditioning produced identical code bytes (1,236 B) and matching audit NRMSE (mean 3.1e-12; max 4.0e-11 on seed 62601, mean 3.1e-11; max 2.54e-10 on 62602). The Mirror wrapper used a higher arithmetic proxy: 68,416–69,056 versus 43,840–44,480. Both shared methods used about 0.52% of independent grid payload bytes (236,666–236,709 B); grid interpolation mean nRMSE was 0.00076–0.00090. Fresh seeds remained sealed after the exact native alias.

**C — Strongest counter-hypothesis.** This is standard coordinate conditioning of a canonical circle SDF. Direct center/radius conditioning reproduces it exactly and computes it more cheaply. The byte reduction is explained by the analytic shared decoder versus independent dense fields, not by a Mirror-specific mechanism.

**U — Unconfirmed.** Only translated/scaled circles were measured. No learned DeepSDF decoder, general shape family, 3D scans, or decoder training was tested. Compression against learned latent auto-decoding remains open.

## Fact / Interpretation / Hypothesis

**Fact:** the shared code payload is ~0.52% of independent dense grid bytes, and the direct analytic conditioner aliases its output while using fewer operations.

**Interpretation:** a shared analytic function plus three values is an efficient representation of this narrow shape family; Mirror adds no demonstrated benefit over ordinary geometric conditioning.

**Hypothesis:** code geometry may matter for broader shape families only if it beats standard latent auto-decoding at equal quality and full payload cost.
