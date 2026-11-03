# MA-621 — CAVIA Mirror task context

Status: FAIL (development screen; fresh sealed)  
Branch: `research/ma-621-cavia-mirror-context-20261009`  
Base commit: `10f2ad2f`  
Prior art: PA129 CAVIA

## H

A one-coordinate Mirror task context adapts faster or uses fewer actual bytes than CAVIA's unconstrained task context while retaining held-out task quality.

## Frozen screen

Sixteen regression tasks use transformations on a shared 8×8 rotation-orbit basis. Each task provides four support examples and 256 held-out examples. Compare shared no-context model, Mirror angle adaptation by a fixed 180-point support-loss grid, CAVIA unconstrained two-coefficient least squares over `[W,JW]`, one-dimensional direct angle context, and independent per-task matrices. Two seeds. Serialize all bases, contexts and metadata; report held-out nMSE, code bytes, adaptation evaluations/steps, MAC proxy and wall time. This tests aligned task-context geometry, not general few-shot learning.

PASS requires Mirror improve held-out quality over shared baseline, beat CAVIA/direct context at matched bytes or adaptation compute, and use <50% independent task bytes. FAIL if direct angle context aliases Mirror. Fresh remains sealed on alias.

## H / T / D / C / U

- **H:** A structured one-coordinate context gives a useful adaptation/storage advantage over unconstrained CAVIA context.
- **T:** 16 tasks, four support examples/task, 256 disjoint query examples/task and two development seeds. Shared 8×8 operator; Mirror angle selected by 180-point grid; CAVIA two-coefficient least squares; byte-identical direct angle; independent task matrices. NPZ inference payloads measured including bases, codes and metadata.
- **D:** FAIL. Mirror improved over shared baseline and was smaller than CAVIA/independent, but direct angle was byte/output-identical. CAVIA reached numerical-zero held-out error using 16 adaptation evaluations vs 2,880 for Mirror. The registered Mirror specificity/adaptation gate failed; fresh remained sealed.
- **C:** CAVIA context, direct two-coefficient basis, independent tasks.
- **U:** Meta-training, nonlinear tasks, task-distribution shifts and learned inner-loop adaptation. Wall clock was not measured. The task family is deliberately aligned and does not establish broad CAVIA performance.

## Fact / interpretation / hypothesis

- **Fact:** Across two seeds Mirror nMSE was 8.67e-5–8.96e-5 at 1,012 B; direct-angle was identical. CAVIA was 1.18e-31/7.81e-32 at 1,553 B with 16 support-fit evaluations, versus 2,880 for the Mirror grid. Independent matrices used 4,575 B and zero error.
- **Interpretation:** A structured one-dimensional task code compresses this aligned orbit, but the gain is not Mirror-specific and its tested adaptation is much more expensive than the simple CAVIA fit.
- **Hypothesis:** With nonlinear/off-orbit task families, CAVIA may require private state while structured views could help; this experiment did not test it.
