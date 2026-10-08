# MA-539 — Packet-level function vector

Status: FAIL (Mirror-specific gate)
Evidence lane: MECHANISM / STORAGE / RUNTIME
Base commit: `28194ea` (`research/mirror-application-worker-ready-20261007`)

## Hypothesis

H: A single compact function vector estimated for a task and shared across all future packet slots can encode slot-coupled functions with a better payload/quality point than a generic PTP-style packet latent, after counting the vector, shared basis, and private exceptions.

## Mirror insertion

> **Mirror insertion:** this experiment adds a task function vector `m_t` to the shared packet decoder's functional coefficient interface so that all P future-slot functions can vary together without storing a separate full weight matrix for every task and slot.

Native method: parallel packet prediction with one generic shared packet latent. `m_t` is a low-dimensional continuous coordinate extracted from support examples and used by every slot. It is persistent per task and its stored code is paid. The nearest ordinary control is the same rank-q generic packet-latent coefficient model; if that matches, there is no Mirror-specific result.

## Prior-art delta

PA10 conditions parallel tokens on sequence randomness; PA99 finds behavior-carrying activation vectors and partial composition. TM001 found hidden packet uncertainty was not solved by factorized phase slots and that a generic packet latent improved but did not close the AR gap. MA-121 already tested phase-slot views of a shared output head. MA-539 isolates a task-level function coordinate shared across the complete packet and directly tests it against generic PTP latent codes plus private exceptions. This is a small linear functional mechanism screen, not a Transformer or language-model claim.

## Controls and task

A shared-context linear packet teacher produces four continuous slot outputs. Six tasks share a rank-two coefficient basis and task-specific two-dimensional codes; two tasks use unrelated per-slot private functions. Support examples estimate each shared task code (or private per-slot weights). Held-out contexts measure normalized packet-output MSE and exact sign-packet accuracy.

Controls:
1. hard-tied shared base (no task code);
2. ordinary PTP shared packet latent with rank-q task coefficients;
3. Mirror function vector `m_t` shared across all four slots, with the same rank-q decoder basis;
4. independent per-task/per-slot linear functions as an upper reference.

The function-vector and PTP parameterizations are deliberately matched in code dimension and function basis. Their serialized payloads include all per-task codes, shared bases, private weights, and metadata. A Mirror-specific pass requires a useful fresh quality/bytes improvement over that control; shared functionality alone is not evidence of new capacity.

## Selection and freeze

Candidate selected by random draw 7 from a 456-item sorted P0/UNTESTED pool: index 133 selected MA-539. The full pool is saved at `source/selection_pool.csv`. Earlier completed IDs and paused families were excluded. No MA-539 research branch was found. Pool SHA-256 is recorded in PROTOCOL.json.

Development seeds 53901–53902 choose the smallest q from {1,2,4} with mean Mirror normalized MSE ≤1e-6, then select ridge penalty from {0, 1e-4, 1e-2} by mean MSE. Fresh seeds 53911–53913 remain sealed until source/config freeze.

## Gates

**PASS:** across all three fresh seeds, Mirror normalized MSE is no worse than 1.05× ordinary PTP; exact sign-packet accuracy is no worse by >0.01; actual serialized bytes are at least 10% below PTP; inference MAC proxy is no more than 1.25× PTP.

**FAIL:** the quality criteria pass but Mirror misses the 10% byte margin against ordinary PTP, or loses the quality gate. This falsifies a Mirror-specific advantage in this interface.

**NOT ESTABLISHED:** serialization/replay or split integrity fails. A pass remains synthetic mechanism evidence only.

## H / T / D / C / U

H: one shared low-dimensional `m_t` may recover task-specific packet functions without per-task/per-slot matrices.

T: pending frozen run; model, support/test budgets, rank selection, controls, seeds, and byte serializer are specified in PROTOCOL.json.

D: FAIL for a Mirror-specific improvement. 3/3 fresh seeds match generic PTP quality exactly, but the Mirror payload is 1,106 B vs 1,041 B and decode MACs are 221,184 vs 196,608.

C: generic PTP latent may be algebraically identical to the function vector and match it at equal code dimension.

U: natural language, a trained Transformer decoder, hidden stochastic packet modes without an exposed code, and end-to-end decoding throughput.

## Fact / Interpretation / Hypothesis

FACT: At q=2 and ridge 0, all 3 fresh seeds give both PTP and FV normalized packet MSE near 1.10e-15 and exact sign-packet accuracy 1.0. Mirror serializes 1,106 B vs PTP 1,041 B (+6.2%) and uses 12.5% more inference MACs.
INTERPRETATION: The packet-level function vector works as a compact task coordinate relative to full task-slot weights, but the generic PTP latent does the same job with fewer bytes and MACs. The preregistered >=10% Mirror storage edge fails.
HYPOTHESIS: In this linear packet interface, FV conditioning is an activation-space parameterization of the same shared packet latent rather than an additional Mirror-specific degree of freedom.
BOUNDARY: Synthetic task-conditioned linear packet outputs only; no language model, nonlinear decoder, or hidden stochastic packet source was tested.
