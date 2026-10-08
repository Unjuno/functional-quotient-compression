# MA-444 — structured Mirror map versus LEO decoder

Status: SCREENING  
Evidence lane: FEW-SHOT / DECODER BYTES / QUALITY  
Base commit: `76327f1`

## H — falsifiable hypothesis

At matched two-dimensional latent size and adaptation budget, a structured Givens Mirror decoder maps task coordinates into useful functions with fewer actual decoder bytes than a generic LEO latent-to-weight basis, while keeping held-out query error within 1.10x.

> **Mirror insertion:** this experiment replaces LEO's generic latent-to-weight map with a structured two-coordinate View of one shared base function.

## PA76 delta

LEO learns a low-dimensional task latent and decodes it to parameters. MA-444 holds latent dimension and adaptation data fixed and tests the marginal value of the structured Mirror map itself against a generic linear decoder, full-vector adaptation, and no adaptation.

## Protocol

Paired 8D Givens-orbit regression as MA-442/443. Same 2D latent and 5 support-gradient updates; report 0/1/3/5 update NRMSE. Meta-train on task IDs 1000..1127, select outer LR {0.01,0.03} on development task IDs 0..11 in worlds 44400/44401. Fresh worlds 44410/44411/44412, seeds 0/1/2, tasks 2000..2019, disjoint from train/dev. Charge shared base, decoder basis, code, and metadata; report N=1/20/100 task amortization.

PASS: Mirror within 1.10x LEO query NRMSE and uses fewer actual amortized decoder/task bytes at N=20 in all fresh worlds. FAIL if quality or bytes gate misses or the generic decoder wins quality at similar bytes.

## C — strongest counter-hypothesis

The Givens map may be too restrictive even for the aligned task, while a generic two-vector decoder spends more bytes but covers more useful directions.

## U — unresolved

Natural tasks, nonlinear network backbones, and support distribution shift remain untested.
