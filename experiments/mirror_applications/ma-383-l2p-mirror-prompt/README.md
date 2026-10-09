# MA-383 — L2P prompt pool with Mirror prompt generator

Status: **FAIL on the frozen development gate; fresh seeds remain sealed.** Dedicated branch: `research/ma-383-l2p-mirror-prompt-20261009`.

## Mirror insertion

**Mirror insertion:** this experiment adds one Givens angle `m_i` to a shared two-vector prompt basis so that eight prompt-conditioned functions can vary without storing eight full 64D prompt values.

## H — Hypothesis

A per-prompt Mirror coordinate can preserve retrieved prompt function quality while storing at most 60% of the explicit prompt/key payload, with the same task-identity-free nearest-key retrieval as L2P.

## T — What was run

PA56/L2P uses a prompt pool retrieved from input features without task identity. This frozen-feature screen trained eight length-64 prompt values decoded as 8×8 residual maps, with eight paid 8D keys. Compared explicit prompts, hard tying, scalar gates, generic two-coordinate basis values and Givens-angle values. All methods used identical noisy queries and nearest-key retrieval. Two development worlds (38301/38302), 1,200 Adam updates, LR .01, batch 64. The protocol was frozen at `18f5646f` before implementation/development.

## D — Decision

**Fact:** Retrieval top-1 was identical across methods because the same keys/queries were used: 89.45% and 89.70%, below the frozen 90% task-validity threshold. Explicit prompt-pool payloads were 3,628/3,627B; Mirror was 2,424/2,423B (66.8% of explicit, above the ≤60% limit). Mirror oracle-key prompt NRMSE was .601 in world 38301 and .000033 in 38302, while explicit prompts were below 1e-7; the Mirror source optimization did not reproduce reliably across development worlds. After retrieval, Mirror NRMSE was .6635/.4251 versus explicit .4567/.4251, generic coordinates .4567/.4251, scalar gates .6679/.6124, and hard tying .9189/.8432. Mirror was also about 12.6% larger than scalar, outside the 1.10x byte margin. Ten payloads pass size/hash/reload/metric replay; three tests pass.

**Interpretation:** The registered system gate failed for retrieval validity, prompt reconstruction reliability, total bytes and scalar byte margin. Since the explicit L2P control itself missed the retrieval-validity gate and retrieval was identical across methods, downstream retrieval quality cannot isolate the effect of prompt compression. The generic coefficient bank also matches or beats Mirror at slightly larger payload.

## C — Strongest counter-hypothesis

The fixed query/key noise made the task invalid for every variant: even the explicit baseline retrieved the wrong prompt about 10.5% of the time. The first Mirror world also suggests an optimization/initialization failure, not a representation limit; the second world reconstructed its prompt orbit well. The preregistered no-tuning rule prevents repairing either issue using these worlds.

## U — Unresolved

No pretrained L2P model, continual-learning benchmark, task-stream forgetting, natural prompt pool, or new retrieval-noise protocol was tested. Fresh worlds 38311–38313 were not opened. This is not evidence about natural prompt capacity.

## Evidence labels

- **Fact:** per-world data and all serialized payloads are under `results/development/`; hashes and replay checks are in `VERIFICATION.json`.
- **Interpretation:** this frozen screen does not establish a usable Mirror prompt-compression point; the baseline retrieval channel and one-world Mirror fitting already fail validity/quality requirements.
- **Hypothesis:** a new task-validity pilot with query noise selected only on development and a stable prompt-basis initialization could test the same storage question under a valid retrieval regime; it would require a new amendment/ID and fresh worlds.
