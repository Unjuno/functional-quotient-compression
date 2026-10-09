# MA-468 — Shared/private skill bank with Mirror skill views

Status: **FAIL under quality and storage gates; private boundary measured**  
Evidence lane: MODULAR SKILLS / LOGICAL MULTIPLICITY / ALLOCATION / BYTES  
A1 protocol freeze: `2f26575d`; valid fresh worlds 46820–46822. Initial 46810–46812 are archived as exploratory because the serialized full-bank skill map was incorrect.

## H — Hypothesis

Two physical skill modules plus per-skill Mirror Views can represent four logical skills and held-out two-skill compositions using materially fewer bytes than a four-module Polytropon-style bank, while retaining quality. A private residual should recover an off-orbit skill if needed.

## T — Test

Four 2D linear skill functions: skills 0/1 were rotated views of module A, skill 2 was module B, and skill 3 was off-orbit B plus a private residual. Evaluated all six pairwise task allocations, including two held-out pairs. Compared two modules without Views, two modules with Views, Views plus a private residual, and four independent modules. Each skill had 64 calibration samples and each task had 256 query samples. A1 corrected the full-bank serialized skill routing map and used fresh replacement worlds. Actual payloads include physical modules, angles, residual, allocations, and metadata.

## D — FAIL; private residual crossover

**Fact:** At N=6 held-out pairs, mean NRMSE was 0.511 for two modules/no Views, 0.375 for Mirror Views, 0.00101 for Mirror plus private residual, and 8.98e-8 for four physical modules. Payload bytes were 2,209B, 2,461B, 2,713B, and 2,209B respectively. Mirror improved over no-view tying, but its package was larger than the four-module control and quality remained far worse. The private residual nearly recovered quality but cost more bytes than the full bank. Mirror view search used 184,832 MAC per skill calibration sequence.

**Interpretation:** Views recover part of the rotated skill family, while the off-orbit skill needs private state. The serialized two-module-plus-angle payload is not smaller than the full bank under actual byte accounting, so the registered compression gate fails. Skill allocation indices were provided equally to all methods; this does not test learned allocation quality.

## C — Strongest counter-hypothesis

The test makes three skills rotation-aligned but deliberately includes one off-orbit skill. The result locates a private-state boundary in this toy bank, not the general utility of Views in learned skill inventories.

## U — Unknown

Natural modular skill banks, learned allocation, higher-dimensional skills, and language/robot task transfer remain untested.
