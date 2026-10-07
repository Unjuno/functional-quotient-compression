# MA-004 — nonlinear soft Mirror expert mixture

Status: **PROMISING** (aligned dense soft-mixture gate passed 3/3; hard tying has a smaller/faster frontier)
Evidence lane: MECHANISM / STORAGE / COMPUTE / RUNTIME
Branch: `research/ma-004-soft-mirror-expert-mixture-20261007`
Base commit: `7ff5430` (verified MA-002 parent)

## H — hypothesis

For a dense softmax MoE whose nonlinear expert functions are related by per-role Givens coordinates, one shared FFN with four charged views can recover the all-expert probability-weighted output at lower serialized bytes than untied MoE. It must outperform hard tying and byte-near FiLM/rank-2 residual controls to support a Mirror-specific claim. Independent role FFNs test the private-parameter boundary.

## Prior-art delta

- MA-001/MA-002 tested nonlinear top-1 and sparse top-2 routing. This experiment sends every token through all four experts and mixes all four outputs with learned softmax weights.
- MA-005 tested an all-expert linear signed mixture with deterministic Walsh coefficients. Here experts are nonlinear, coefficients are learned input-conditioned router probabilities, and all role contributions are positive and normalized.
- PA01 requires hard expert tying; PA02/PA03 motivate router sharing/factorization controls. This single-layer screen does not test path constraints.

## Physical-to-logical claim

One 16→32→16 GELU FFN plus four role-specific four-angle Givens views creates four router-addressable expert functions. Each example composes all four with softmax probabilities. Controls are full untied experts, rank-2 factorized-router full experts, hard tying, tied+FiLM, tied+rank-2 residual, and tied+Mirror.

## T — protocol

See `PROTOCOL.json`. A frozen teacher linear router provides a full softmax distribution; the learned student router is distilled on all four probabilities. The aligned teacher uses Givens-conjugated views of one nonlinear FFN; independent mode uses four unrelated FFNs. Development world 40000 selects one LR. Fresh worlds 40001–40003 are opened only after the preregistered quality/byte gate passes.

## Storage and compute

Measure actual serialized state-dict plus deterministic config bytes, charging router, all experts or shared FFN, views/controls, and metadata. Report four-active-expert MAC proxy, router/coordinate operations, updates/examples, wall time, and CPU throughput.

## Results

Development selected LR 0.01 by the preregistered mean-MSE rule. The development aligned Mirror/dense MSE ratio was 0.385 and payload ratio 0.331, passing the frozen fresh-access gate. Fresh worlds 40001–40003 then ran without changes. All 36 deterministic fresh rows replayed exactly.

### Fact

- Aligned Mirror MSE was 0.0000694 / 0.0001362 / 0.0001998, or 0.297x / 0.487x / 1.030x dense untied soft MoE. Dominant-role accuracy was 0.05 percentage points above the dense control in all three worlds. The registered useful-sharing quality/storage gate passed 3/3.
- Mirror payload was 7,697B vs 23,277B dense soft MoE (0.331x) and 23,466B rank-2-router MoE (0.328x). Hard tying used 7,445B, 252B fewer; FiLM and rank-2 residual each used 8,845B.
- Mirror MSE was 0.107–0.290x hard tying, 0.108–0.413x FiLM, and 0.115–0.360x rank-2 residual. The Mirror-specific Pareto gate nevertheless fails vs hard tying: all four soft expert contributions require 4,160 active MACs plus 192 coordinate FLOPs, compared with one tied FFN at 1,088 MACs. Against dense untied MoE, Mirror adds only the 192 coordinate FLOPs to the same four-active-expert proxy.
- Independent role functions did not fit the shared view: Mirror MSE was 0.00165–0.00180 vs 0.000261–0.000316 dense MoE; FiLM and residual controls were also well below Mirror. This indicates unrelated functions require private or richer state.
- Runtime regressed in the eager CPU implementation. Median aligned training wall time was 9.70s for Mirror vs 1.21s hard tying. Median inference throughput was 0.362M vs 4.215M examples/s (0.086x). This single-host timing is implementation-specific.
- Three tests passed; serialization output round-trip was checked on all 72 runs. All 36 deterministic fresh rows replayed exactly; wall time and throughput were excluded from exact replay comparison.

### T — execution

Six methods (dense untied, PA03-style rank-2 router, hard tying, tied+FiLM, tied+rank-2 residual, tied+Mirror) trained for 1,200 updates × 64 examples with matched data in aligned and independent teacher modes. Development world 40000 selected LR 0.01. Fresh worlds 40001–40003 used frozen teacher/data/init seeds. A frozen linear router supplies full softmax probabilities, and each example composes all four nonlinear expert outputs. Serialized bytes charge all router/expert/view/control state and metadata.

### D — decision: PROMISING

The useful-sharing gate passed on a deliberately Givens-aligned teacher, including the boundary seed at 1.03x dense-MoE MSE, while payload was 66.9% smaller than dense MoE. The shared view also strongly improved quality over hard tying, FiLM, and rank-2 residual. It does not produce a Mirror-specific Pareto improvement over hard tying: the tied model is 252B smaller and evaluates one FFN rather than four. Eager CPU throughput is about 11.6x lower than tying. Independent expert functions need private/richer parameters.

### C — strongest counter-hypothesis

The aligned teacher was generated from the same Givens-conjugacy family used by Mirror, and the fixed update budget favors its inductive bias. A generic shared-basis or generated expert control matched for bytes and compute could close the quality gap. Full soft routing also evaluates every role, so it is not a sparse-compute win.

### U — unconfirmed

Natural-language MoE quality, near-convergence or fixed-byte capacity, generic byte-matched basis controls, larger role pools, load-balance behavior at scale, and optimized GPU kernels remain untested.

## Fact / interpretation / hypothesis

- **Fact:** measured values and gate outcomes are in `RESULTS_CORE.csv`; storage uses serialized payload bytes.
- **Interpretation:** a dense learned mixture of four deliberately related nonlinear experts can be represented by one FFN plus small views with much lower payload than untied experts, but with a large runtime cost and no Pareto win over hard tying.
- **Hypothesis:** naturally learned experts may share low-description coordinates; this experiment does not establish that in language models.
