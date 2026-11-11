# MA-849 — context-generated plasticity rule coordinate

H: a compact coordinate on a shared gradient-history-conditioned update rule improves few-step adaptation over a fixed rule across task curvature regimes, with a Mirror-specific advantage over a generic native context-to-rate gate.

PA228 motivates dynamic learned plasticity; PA150 motivates learned optimizer behavior; both require comparison against ordinary context/optimizer rules. Here tasks are synthetic positive-definite quadratic objectives with varying curvature. The online state sees only parameter and gradient history. A secant curvature estimate sets the rate; a scalar coordinate modulates it. The native gate uses the same scalar degree of freedom directly.

Development seed 17 tunes a fixed rate and scalar multiplier; fresh seeds 37/59 are frozen. Twelve updates per task. Count rule bytes, writable state, updates, active MAC proxies, wall-clock, and final excess loss. This is a mechanism screen, not a trained learned-optimizer result.
