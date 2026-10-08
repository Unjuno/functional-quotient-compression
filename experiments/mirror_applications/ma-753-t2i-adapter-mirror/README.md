# MA-753 — Mirror compression for composable image-latent controls

Status: SCREENING
Base commit: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`
Prior art: PA191 — T2I-Adapter

## H — falsifiable hypothesis

A shared adapter matrix with a small control-specific Givens coordinate can represent multiple aligned control adapters and held-out additive compositions with less actual serialized state than independent adapters. Compare with an equal-size unrestricted basis mixture before making a Mirror-specific claim.

## T — frozen scope

This first screen isolates the adapter mechanism in a synthetic 8D latent regression because the available container is CPU-only and no pretrained diffusion checkpoint is provisioned. Four control types are trained individually; all six two-control sums are held out for audit. The aligned world uses four Givens views of one matrix; the boundary world uses four independent matrices. Controls are independent adapters, an equal-size two-matrix basis mixture, per-control FiLM, and hard sharing. Exact conditions/seeds/gates are in `PROTOCOL.json`; Draw30 replay is in `source/draw30_exclusions.json`.

## D — decision

Pending development freeze and fresh audit.

## C — strongest counter-hypothesis

The teacher is explicitly Givens-aligned, so any Mirror success may be a construction advantage. Independent adapters or a plain basis mixture may better preserve useful compositions for unrelated controls; eager Givens operations may add latency.

## U — boundaries

This experiment cannot establish image quality or T2I performance. A positive result applies only to this fixed-feature composition mechanism screen.
