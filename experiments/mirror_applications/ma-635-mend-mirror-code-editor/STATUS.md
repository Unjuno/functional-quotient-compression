# MA-635 status

- Status: FAIL (fixed-budget development quality gate and native specificity)
- Branch: `research/ma-635-mend-mirror-code-editor-20261103`
- Base commit: `bb89e261`
- Protocol frozen before development: yes
- Development complete: yes (63501, 63502)
- Fresh/audit opened: no; quality gate missed and native low-rank editor aliases Mirror
- Results committed: pending
- Verification committed: pending

## H / T / D / C / U

**H:** A compact code-predicting editor preserves held-out edit efficacy with less payload and compute than generic MEND; Mirror must beat native low-rank MEND.

**T:** Synthetic 8D gradient features map to rank-2 32D update vectors. Trained generic 32-output and 2-code MLP editors with the same 32-unit hidden layer, Adam, 1,000 updates, batch size 64 and 2,048 training rows. Evaluated 256 audit rows on development seeds 63501/63502. Native low-rank control copied the same trained code editor and decoder. Complete serialized payloads and active operations measured.

**D — FAIL.** Generic MEND update nRMSE was 0.0533 and 0.0579, just above the frozen ≤0.05 gate, at 9,688–9,705 B and 327,680 operation proxy. Mirror code editor nRMSE was 0.0681 and 0.0619, at 6,505–6,511 B and 98,304 operations. Thus the code model saved about 33% bytes and 70% operations but missed the quality gate. Native low-rank MEND exactly matched Mirror arrays, outputs, bytes and operation count. Fresh stayed sealed.

**C — Strongest counter-hypothesis.** The compact editor is ordinary low-rank MEND. Its lower compute follows from emitting two coefficients instead of 32 update values; it did not preserve the predeclared edit efficacy threshold at the fixed update budget.

**U — Unconfirmed.** This is a fixed-budget synthetic learning-efficiency screen, not a capacity result. No near-convergence run, real ROME/MEND editor, factuality, or model locality task was tested.

## Fact / Interpretation / Hypothesis

**Fact:** code emission reduced serialized bytes and active operation proxy relative to the generic output head, but increased held-out nRMSE above 0.05. Native low-rank MEND was an exact alias.

**Interpretation:** lower-dimensional editor outputs can reduce cost, but this setup did not retain enough fixed-budget quality and showed no Mirror-specific effect.

**Hypothesis:** a trained editor may benefit from code supervision or a better shared basis, but any such change needs a new preregistered experiment and real edit evaluations.
