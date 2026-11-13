# MA-969 results — invalid implementation screen

Status: **NOT ESTABLISHED**. The registered sparse-private-synapse mechanism was not implemented as specified, and the shared D1 core and independent D2 control were near chance. Preserve this run as a diagnostic artifact; it is not evidence for or against Mirror/private-state capacity.

## H — hypothesis

A D1-trained shared recurrent SNN could retain D1 performance and adapt to D2 with a compact threshold coordinate and sparse private synapse residuals, matching an independent D2 SNN at lower actual joint payload bytes.

## T — execution

**Fact.** The screen used public D1/D2 Vernier event records (708 trials per distance label set), first 48 chronological events at/after 80 ms, a 8-neuron recurrent LIF, one stratified whole-trial split, one development optimizer seed (96901), and three fresh optimizer seeds (96911, 96913, 96917). All fresh optimizer seeds share one acquisition and one held-out test partition. Controls were zero-shot shared core, direct thresholds, a one-dimensional Mirror threshold coordinate, Mirror plus top-k synapse entries (k=1,2,4,8,16,32), and an independent D2 SNN. Actual `torch.save` joint payloads and per-condition evaluation metrics were recorded in `RESULTS_CORE.csv`.

**Fact.** The D1 fresh accuracy was 47.3%, 47.3%, and 52.7% across seeds. Independent D2 fresh accuracy was 52.0%, 50.7%, and 52.7%. These scores do not establish useful D1 learning or D2 task adaptation.

**Fact.** Astra code review found a critical implementation mismatch: `scatter(0, mask, delta)` replaces selected shared weights with zero-initialized delta values instead of adding residuals to shared weights. For k=32 this zeros every selected input synapse at initialization. The frozen protocol specifies a residual (`shared + delta`), so the sparse mechanism was not tested. The spike backward derivative also differs from the protocol's stated sigmoid surrogate, and `active_mac_proxy` charges k extra products although the implementation performs a dense matrix product and no k-entry residual multiply.

**Fact.** Serialization replay is incomplete: the replay path reconstructs adapters from the in-memory D1 base rather than the decoded `shared_core`, and zero-shot does not replay. Byte counts are therefore raw measured archive sizes, not validated end-to-end inference replay for every condition.

## D — decision

**NOT ESTABLISHED.** Do not classify this as a scientific FAIL of Mirror or sparse residuals: the core residual operation contradicts the frozen protocol, the compute proxy does not match executed operations, and the shared/independent quality controls are near chance. Preserve the preregistered success-gate miss as an observed property of this invalid run only. Correcting the code requires an explicit amendment or new experiment ID and fresh audit data.

## C — strongest counter-hypothesis

**Interpretation.** The apparent failure may be caused by an unlearned/underpowered D1 recurrent core and by replacing rather than adding private synaptic corrections. It does not establish that task-specific synapses are necessary or that Mirror views cannot recover the task.

## U — unresolved

- Whether a correctly implemented residual plus threshold view can adapt between D1 and D2 while retaining D1 quality.
- Whether the tiny from-scratch LIF can learn the real Vernier sign task with a useful independent control.
- Whether serialized decoded state reproduces logits for every condition.
- Whether an operation-matched MAC/latency/energy frontier favors sparse residuals.
- Generalization across independent acquisitions; this run has one acquisition only.

## Evidence separation

- **Fact:** frozen source, recorded fresh scores, byte counts, code review findings, and protocol mismatches above.
- **Interpretation:** this implementation/run cannot answer the registered private-state question.
- **Hypothesis:** a corrected residual implementation and a learnable matched baseline could resolve the boundary, but requires a new protocol boundary.
