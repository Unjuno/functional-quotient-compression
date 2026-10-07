# MA-251 — factorized expert x depth Mirror coordinate

Status: SCREENING  
Evidence lane: MECHANISM / STORAGE / RUNTIME  
Base commit: `809f9d50ef85d470d8bac045db291349e264dcba`

## Hypothesis

Factorized expert and depth Mirror coordinates can express a Cartesian family of logical role-depth functions from one shared expert matrix with fewer actual bytes than a per-pair coordinate table, when teacher roles factor across expert and depth. Independent pair functions should expose the need for private/table parameters.

## Prior art delta

PA01 shares expert weights across depth and requires ordinary expert tying as a control. PA06 reuses blocks with per-step generated modulation and motivates static per-depth LoRA controls. MA-251 tests a product coordinate `(expert, depth)` while explicitly measuring whether the resulting logical combinations correspond to task quality rather than counting combinations as capacity.

## Task

A 16D input is mapped to 12D output for one of 4 expert identities and 4 depth positions. The aligned teacher applies expert Givens rotations to one half of the hidden coordinates and depth Givens rotations to the other half before one shared matrix. The negative teacher uses an independent matrix for every expert-depth pair. Controls span full Cartesian parameters, one-axis tying, static depth LoRA, single-axis views, factorized views and an explicit per-pair coordinate table.

## Development screen

World 25100 selected LR 0.01 by mean MSE across nine methods and both modes (2.7091 vs 2.7123 at LR 0.003). On the factorized teacher, factorized Mirror MSE was 1.25e-9 at 3,608B; Cartesian Mirror was 1.55e-9 at 3,803B; untied reached 1.29e-8 at 15,317B. Hard tying, expert/depth one-axis views, expert tying, depth tying and static depth LoRA had MSE 0.19–0.40. On independent pairs, factorized Mirror MSE was 5.28 and the Cartesian table 4.81, while untied reached 1.02e-8. Fresh worlds 25101–25103 remain unopened.

## Decision

FACT: pending.  
INTERPRETATION: pending.  
HYPOTHESIS: pending.  
BOUNDARY: pending.
