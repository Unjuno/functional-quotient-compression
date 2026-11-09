# MA-761 — Goal coordinate for one shared NCA-inspired local rule

Status: **FAIL** (scoped synthetic Mirror-specific claim)

## Fact

On five held-out goal cells, a shared coordinate-conditioned rule reaches 100% rollout success with mean 2.6 steps at 5,631 B and 291,840 MACs. A bank of 20 independent per-goal policies occupies 118,319 B but has no policies for the held-out goals (0% success/0 of 5 coverage). A shared rule without goal context and a shared one-hot goal embedding each reach 40% success. Native coordinate-conditioned and Mirror payloads, hashes, rollouts and MACs are identical.

## Interpretation

The 2D goal coordinate provides a compact, interpolating functional input for this NCA-inspired navigation rule; it generalizes to goal locations absent from training and is much smaller than an independent policy bank. Ordinary coordinate conditioning fully explains that behavior. This is a useful coordinate-conditioning signal, not a Mirror-specific gain or evidence for general NCA pattern growth.

## Hypothesis result

**FAIL** for Mirror-specific value because the native coordinate-conditioned control is an exact function/payload alias. The generic coordinate-conditioned rule passes the synthetic held-out rollout and byte comparisons. Fresh/audit was not opened after exact alias.

## Frozen protocol / scope

PA195/PA196 motivate a shared recurrent local update rule that behaves differently under goal context. Here, a single active cell moves over a 5×5 obstacle-free grid. A shared MLP local transition predicts one-step movement from current position and goal. Twenty goals train the shared rule; five goal cells are held out entirely. Four corner starts per held-out goal measure rollout success and steps. Compare independent per-goal policies for seen goals, hard tying without a goal code, a shared one-hot goal embedding, native continuous coordinate conditioning, and Mirror coordinate views. Full NPZ payload bytes, hashes, optimizer updates, MAC proxy and wall time are recorded.

This is a single-agent navigation screen with an NCA-inspired shared cell rule. It does not test growing/regenerating patterns, multi-cell interactions, obstacles, morphology, biological realism or a trained GoalNCA benchmark.

## H / T / D / C / U

- **H:** A small goal coordinate lets one physical local rule reach unseen goal behaviors with at least 25% fewer complete bytes than an independent rule bank.
- **T:** Seed 76101; 5×5 grid; 20 training goals; 5 heldout goals; 4 corner starts each; shared 4→32→32→2 rule; 900 shared updates and 400 per observed independent rule.
- **D:** **FAIL** for Mirror attribution. Coordinate conditioning reaches all heldout goals at 5,631 B; native coordinate code exactly aliases Mirror. Independent bank uses 118,319 B but cannot cover unseen IDs.
- **C:** Standard continuous coordinate conditioning explains the result; the independent ID bank is deliberately unable to extrapolate to IDs it never trained.
- **U:** Real GoalNCA tasks, obstacles, patterned growth/regeneration, robust dynamics, broader unseen-goal distributions and accelerator cost.
