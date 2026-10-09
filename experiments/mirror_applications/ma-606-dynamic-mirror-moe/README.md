# MA-606 — Dynamic Mirror-MoE without discrete expert routing

Status: SCREENING
Evidence lane: MECHANISM / STORAGE / COMPUTE
Base commit: `9fc15d0b` (worker-ready baseline plus MA-602–604 evidence)
Prior art: PA121 CondConv; soft-MoE output mixing is the principal control

## Hypothesis

H: A small input-conditioned Mirror coordinate that synthesizes one logical FFN per input can approach a standard soft-MoE's held-out quality while using substantially fewer actual bytes and active FFN compute, and can outperform a parameter-near FiLM control.

## Mirror insertion

> **Mirror insertion:** this experiment adds a three-angle input-conditioned Givens coordinate `m(x)` to the shared FFN's hidden preactivation weights, so each input receives a distinct logical FFN view without storing or evaluating four independent experts.

The main comparison is standard soft-MoE (four expert outputs evaluated and mixed), one shared FFN + FiLM, and one shared FFN + Givens View. The teacher is a four-expert soft-MoE with experts sharing a nearby initialization plus fixed private perturbations.

## Gates

PASS requires Mirror normalized MSE <=1.25× soft-MoE and <=1.15× FiLM in both development worlds, payload <=0.60× soft-MoE, and compute proxy <=0.50× soft-MoE. Fresh runs only if every development condition passes. Otherwise FAIL and fresh remains sealed.

## H / T / D / C / U

**H — hypothesis:** a small input-conditioned Givens coordinate can synthesize one logical FFN per example and approximate a soft-MoE with substantially lower storage and active FFN compute, while improving on a parameter-near FiLM control.

**T — execution:** CPU PyTorch 2.14.1; four-expert 16→12→10 soft-MoE teacher, whose expert weights share a common random initialization plus fixed independent perturbations. Development worlds 60601/60602; 4,096 train and 1,024 held-out examples; 1,200 AdamW updates. Controls were static shared FFN, single FFN + three-angle hidden Givens View, single FFN + generated FiLM, trained four-expert soft-MoE, and privileged oracle. Fresh 60611/60612 were not opened after the development gate miss.

**D — FAIL:** Mirror normalized MSE was 0.1830/0.2113, versus FiLM 0.1231/0.1322 and soft-MoE 0.01365/0.01881. It used 4,245 B and 792 MAC proxy versus soft-MoE's 8,277 B and 2,624 proxy (49% fewer bytes, 70% lower proxy), but quality was 11.2×/13.4× worse than soft-MoE and 1.49×/1.60× worse than FiLM. CPU training runtime was 1.21/1.21 s for Mirror, 1.17/1.16 s for soft-MoE and 0.62/0.59 s for FiLM. Frozen quality gates failed; fresh remained sealed.

**C — strongest counter-hypothesis:** the teacher's output is a mixture of four nonlinear expert functions. Rotating one shared hidden preactivation cannot recover that function family; even unconstrained FiLM modulation is a better single-FFN control. The MAC proxy savings do not translate to CPU wall-clock savings because the small Givens path has overhead.

**U — boundaries:** synthetic soft-MoE teacher, fixed update budget and CPU proxy/runtime only. This does not establish language-model quality, converged capacity, or a general MoE compression ratio.

## Facts / interpretation / hypothesis

- **Fact:** all ten actual serialized payloads matched recorded sizes and hashes and reloaded to identical outputs; two tests passed.
- **Interpretation:** reducing active expert computation by synthesizing a single structured view trades away too much useful function on this expert mixture. Mirror is also inferior to simpler FiLM for quality and measured CPU runtime.
- **Hypothesis:** one-view synthesis may fit tasks where expert functions lie on a low-dimensional transform orbit, but independent expert nonlinearities need more private or compositional state.
