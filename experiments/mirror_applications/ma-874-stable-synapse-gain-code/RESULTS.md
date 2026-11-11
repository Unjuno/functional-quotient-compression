# MA-874 results: stable recurrent synapses and gain context

## H — Hypothesis

A two-scalar context code decoded into gains over a fixed recurrent reservoir should implement heldout temporal filters with less per-task stored state and fewer support sequences than writable recurrent-decay updates, while preserving reset behavior. Mirror-specific attribution requires improvement over native PCA/context-gain state.

## T — Trial

The synthetic PA235-inspired reservoir had eight fixed diagonal recurrent channels `h_k[t] = lambda_k h_k[t-1] + x[t]`. Each task's target readout was generated from an independent two-coordinate code over a world-specific rank-2 basis. Sixteen training tasks estimated the basis; sixteen heldout tasks used independent codes. Fresh worlds were seeds 983, 997, and 1009; each had eight support sequences of length 64 available for the 1/2/4/8-shot sweep and eight heldout test sequences per task. Controls were native PCA/context-gain, independent eight-coefficient readouts, pooled readout, task-specific recurrent-decay updates plus scalar gain, and teacher coefficients. The fast-weight optimizer was limited to 30 function evaluations per heldout task and shot count.

## D — Decision

**FAIL for Mirror-specific attribution; frozen-reservoir gain state passes the narrow synthetic quality/reset/storage screen.** One support sequence met the quality threshold in all fresh worlds, and the charged 16-task gain payload was 264 B versus 648 B for the recurrent fast-weight state and 584 B for independent readouts. Native PCA/context-gain exactly matches Mirror function and bytes.

## C — Strongest counter-hypothesis

A standard native output-gain vector over a fixed reservoir explains the result. It exactly duplicates the Mirror function, state, and operations. The fast-weight control is also narrow and ill-matched to this teacher: it changes recurrent decays with a common output gain, while the teacher changes readout mixtures.

## U — Unconfirmed

Trained recurrent networks, nonlinear task dynamics outside a readout subspace, biological mechanisms, learned in-context adaptation, Hebbian/DeltaNet fast weights, long-horizon stability, and real streaming workloads remain untested. This is not a reproduction of PA235.

## Facts

- At one support sequence, mean heldout function MSE was 0.000165 across fresh worlds; worst world mean was 0.000367 and the worst single heldout task was 0.002232, below the frozen 0.01 gate.
- At two support sequences, Mirror/native mean MSE was 0.000151; the fast recurrent-decay control was 0.08793 and did not approach the quality threshold. Independent full readout mean MSE was 0.0000335.
- Actual 16-task inference payloads were 264 B for rank-2 gain codes, 584 B for independent readouts, 648 B for fast-decay vectors plus scalar gains, and 104 B for pooled coefficients. Gain state saves 59.3% versus fast-weight state and 54.8% versus independent readouts.
- Mirror and native context-gain outputs, payload sizes, and decoder proxies match exactly in every fresh world.
- Resetting the reservoir to zero between sequences produced a maximum replay difference of 8.9e-16; no cross-task state contamination was observed in the implemented reset path.
- Reservoir update proxy is 24 operations per timestep. Rank-2 decode proxy is 48 MACs versus 16 for direct readout. Mean rank-2 basis-fit wall time was 0.0031 s. At two shots, the optimizer used about 0.73 s per 16-task batch for fast weights versus 0.00135 s for Mirror adaptation; local timing is host-specific. Maximum full screen wall time was 3.16 s.

## Interpretation

A low-dimensional gain state is a compact way to select temporal responses already present in a fixed reservoir. In this generated family it uses less state than the tested writable-decay updates and preserves a clean reset boundary. It does not show that gain state replaces general weight learning: the fast-weight control cannot express the teacher's readout mixtures, and a native gain vector matches exactly. The rank-2 decode also increases per-query arithmetic.

## Hypothesis for follow-up

A stronger follow-up should use a trained recurrent task family where both stable-synapse context and a properly matched Hebbian/DeltaNet update can learn the same target behavior from identical sequences. It should compare adaptation quality, writable state, reset leakage, and inference compute without restricting the fast-weight method to decay-only changes.
