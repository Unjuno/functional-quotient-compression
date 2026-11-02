# MA-526 status

- Status: NOT ESTABLISHED (execution / implementation screen interrupted)
- Branch: `research/ma-526-sae-feature-mirror-atoms-20261009`
- Prior art: PA102, SAE feature steering
- Fresh/audit opened: yes, during an incomplete run; do not treat these IDs as sealed for confirmation.
- Results committed: no
- Verification committed: no
- Registry row updated: no

## H / T / D / C / U

- **H:** A shared SAE feature bank plus small per-topic Givens coordinates can preserve topic steering with fewer actual code bytes than independent SAE feature lists and a generic shared low-rank control.
- **T:** Pinned GPT-2-small and layer-8 SAE artifacts were fetched. The original 32/16/32-per-class gradient screen exceeded practical CPU time; before accessing fresh identities, the protocol was amended to 8/4/8 per class. Train/dev gradients and frozen selection completed at K=4, alpha=0.25. Exact dev audit completed. The fresh command ran for over eight CPU minutes without producing a results artifact and was stopped.
- **D:** NOT ESTABLISHED. No fresh quality or actual intervention payload comparison is available.
- **C:** The strongest counter-hypothesis is that ordinary sparse/generic coefficients already reproduce the small Mirror code at K=4; exact frozen dev uplifts were identical across all three methods for all four labels. The feature-code path also needs a semantics audit before further results can be trusted.
- **U:** Fresh metrics, serialization comparison, SAE activation/decoder correctness, complete multi-token log-probability and KL scoring, replay tests, and a practical batched inference implementation remain outstanding.

## Facts

- Pinned model/SAE files were downloadable; the dataset revision matched and parquet files were readable.
- Worker was CPU-only (`torch 2.14.1+cpu`, no CUDA). Full-model gradient and per-example forward evaluation was slow.
- Development freeze chose K=4, alpha=0.25. Exact dev uplifts were equal for independent, generic, and Mirror methods on each topic.
- No `FRESH_RESULTS.json` was produced.

## Interpretation

This run did not establish whether the SAE feature-bank Mirror improves the storage/quality frontier. Development gives no sign of Mirror-specific value at the selected setting, but the incomplete low-power screen cannot support a fresh-data FAIL.

## Redesign hypothesis

Before retrying, batch evaluation by topic, verify SAE-Lens activation/decoder semantics against a pinned reference implementation, fix full sequence scoring, and add interruption-safe per-topic checkpoints. Do not reuse opened fresh identities for a confirmatory audit.
