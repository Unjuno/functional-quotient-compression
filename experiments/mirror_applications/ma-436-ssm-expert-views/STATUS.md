# MA-436 status

Status: SCREENING — protocol revision 2 frozen before canonical rerun; initial unamended run is invalid and retained.

## H
Givens state views over a single physical SSM create diverse routed logical experts with a smaller payload than independent SSM banks.

## T
Eight experts, 8-state recurrence, 24 identical sequences/expert, length 128, two development worlds. Givens view affects transition A only; B/C remain shared. Controls: shared no-view, native materialized transition weights, and full independent experts; no training.

## D
Not determined.

## C
The native generated-weight control materializes exactly the same A/B/C matrices from the same four-angle code, so any difference may be an implementation detail rather than a new function.

## U
Actual bytes, output diversity, FP16 recurrence error, and tokens/s. Fresh remains sealed.

## Amendment
Initial code used different inputs per expert and transformed A/B/C together, so its diversity metric was invalid and its view was only a state-coordinate gauge change. That full run and original protocol/hash are retained in `runs/initial_invalid_similarity_gauge/`. Amendment 1 changes to shared inputs and A-only views; the same development seeds are rerun. Fresh remains sealed.
