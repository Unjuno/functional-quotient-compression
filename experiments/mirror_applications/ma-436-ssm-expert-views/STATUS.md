# MA-436 status

Status: **FAIL — amended development screen verified; fresh seeds sealed.**

## H
Givens state views over a single physical SSM create diverse routed logical experts with a smaller payload than independent SSM banks.

## T
Eight experts, 8-state recurrence, 24 identical sequences/expert, length 128, two development worlds. Givens view affects transition A only; B/C remain shared. Controls: shared no-view, native materialized transition weights, and full independent experts; no training.

## D
On the amended same-input task, role-view NRMSE was 0.000434/0.000739 versus 0.222/0.351 for the no-view shared field. Actual payload was 1,809/1,808 bytes versus 2,435/2,450 for independent copies (0.743/0.738x). However, the native generated-A control was byte/hash/output-identical. Throughput was 0.382/0.645x native and diversity RMS 0.0206/0.0252, below the frozen 0.05 threshold. FAIL; fresh remains sealed.

## C
The exact native generated-A control has the same paid state and output. The first screen was invalid for diversity because inputs differed and a full A/B/C gauge transform preserves transfer behavior; protocol amendment 1 corrects both, and those initial artifacts remain excluded.

## U
Trained S4/Mamba quality, natural sequences, longer context, optimized recurrent kernels, and generalization to non-Givens expert banks. No model fitting occurred.

## Amendment
Initial code used different inputs per expert and transformed A/B/C together, so its diversity metric was invalid and its view was only a state-coordinate gauge change. That full run and original protocol/hash are retained in `runs/initial_invalid_similarity_gauge/`. Amendment 1 changed to shared inputs and A-only views; the same development seeds were rerun and canonical replay passed. Fresh remains sealed. The SSM role family is paused after MA-434 and MA-436 repeated exact native conditioning aliases; see [family diagnostic](../../../docs/phase2/STATE_SPACE_MIRROR_FAMILY_DIAGNOSTIC_2026-10-08.md).
