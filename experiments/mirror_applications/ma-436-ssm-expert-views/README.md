# MA-436 — routed logical SSM experts

Status: **FAIL under the amended frozen gate; canonical development screen verified, fresh sealed.** PA73/PA74 reviewed.

## H — Hypothesis
Small routed state-view codes can create several recurrent experts from one physical structured SSM while using fewer actual bytes than full expert copies.

## T — Frozen mechanism screen
Eight experts apply four Givens angles to the state basis around one stable 8-state SSM. Each expert receives 24 sequences of length 128. Compare unconditioned shared SSM, native generated weights, and independent full SSM copies. All routing IDs and reconstruction state are charged. No training occurs; this is a structured synthetic recurrence check, not S4/Mamba or language-model evidence. Fresh seeds 43611–43613 remain sealed.

See `PROTOCOL.json` and `PROTOCOL_AMENDMENT_1.json` for the amended recurrence, input sharing, seeds, payload rules and gates. The initial unamended screen is retained under `runs/initial_invalid_similarity_gauge/` and is invalid because inputs differed across experts and a full A/B/C similarity transform preserves the input-output transfer function.

## D — Amended result
With identical inputs across experts, output NRMSE was **0.000434 / 0.000739** for transition views and **0.222 / 0.351** for the shared no-view SSM. Diversity RMS was **0.0206 / 0.0252**, below the frozen 0.05 threshold. Actual payload was **1,809 / 1,808 bytes** versus **2,435 / 2,450 bytes** for full per-expert copies (0.743 / 0.738x). Throughput was **1.37M / 2.33M** tokens/s versus **3.58M / 3.61M** for the native generated-A control (0.382 / 0.645x).

The native generated-A control used byte-identical payload hashes and replay-identical outputs. Therefore the Mirror-specific gate failed even though the nominal independent-copy byte threshold passed. Canonical output and payload replay passed; no fresh seeds were accessed.

## Execution amendment and provenance
The initial unamended run incorrectly used different inputs for each expert and transformed A/B/C together. Cross-expert diversity was invalid, and the full similarity transform preserved the input-output function. The complete initial run and original protocol hash remain under `runs/initial_invalid_similarity_gauge/`. Amendment 1 broadcasts identical inputs and views transition A only with shared B/C; it reused the same development seeds and frozen thresholds. Canonical amended replay passed.

## C — Strongest counter-hypothesis
The native generated-A control is exactly the same paid transition parameterization. The Mirror's state rotations are also substantially slower in this eager CPU implementation.

## U — Still unconfirmed
Trained S4/Mamba, natural sequence quality, longer contexts, optimized kernels, and non-Givens expert banks. This does not establish language-model or general SSM capacity.

**Evidence labels:** same-input outputs, payloads, diversity and throughput are facts; FAIL follows the frozen alias/runtime/diversity gate; other state-dependent views are hypotheses.
