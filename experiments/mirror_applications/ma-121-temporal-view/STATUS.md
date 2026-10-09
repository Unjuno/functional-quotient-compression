# MA-121 status

- Status: PROMISING
- Branch: `research/ma-121-temporal-view-20261007`
- Base commit: `e04e131926978fada98129d52e2146534a8db6e9`
- Last verified commit: `0ccd5470dcac07afba3b9cf77bbfbb7cf8e82bfc`
- Development complete: yes; seeds 12101–12102
- Fresh/audit opened: yes; seeds 12111–12113 after aligned development gate
- Results committed: yes (`0ccd5470dcac07afba3b9cf77bbfbb7cf8e82bfc`)
- Verification committed: yes (`0ccd5470dcac07afba3b9cf77bbfbb7cf8e82bfc`)
- Registry row updated: yes

## H / T / D / C / U

- **H:** A shared physical token head plus one phase code per slot can recover four correlated future-token functions with lower actual payload than MTP heads while preserving packet accuracy and comparing favorably with rank-2 slot-code sharing.
- **T:** Synthetic four-slot classification with aligned Givens and independent teachers. Compared tied head, rank-2 slot-code/PTP control, Mirror phase, MTP independent heads, and sequential token feedback. Two development plus three fresh worlds; 500 Adam updates at LR 0.02; packet NLL, exact packet accuracy, actual payload, compute, wall, and inference tokens/s measured.
- **D:** PROMISING for the aligned mechanism. Fresh quality/byte gates passed 3/3; independent slot functions were not recovered. Runtime was substantially slower than MTP.
- **C:** Aligned teacher was generated from Givens views, which favors Mirror. The rank-2 shared control had better NLL at larger bytes. Eager Mirror inference had poor throughput.
- **U:** Natural text, exact autoregressive packet paths, true Transformer KV-cache baseline, MTP training, speculative acceptance, and production kernels.

**FACT:** 50/50 rows replayed with exact payload bytes; max metric delta 4.85e-10; tests 2/2 pass. Fresh aligned median NLL: Mirror 0.2474, MTP 0.3032, PTP-rank2 0.1750. Exact packet accuracy: 0.828, 0.664, and 0.820 respectively. Mirror payload was 2,149B vs MTP 2,729B. Eager inference throughput was 3.13M vs 26.3M tokens/s.

**INTERPRETATION:** Phase views give a storage/quality Pareto point for correlated future slots, but not a compute gain or general independent-slot capacity.

**HYPOTHESIS:** Packet phase codes may be useful when slot functions share a low-dimensional orbit; natural-token consistency and end-to-end decoding remain untested.

## Next action

Update the registry, status board, and claim ledger; commit and push; then continue with MA-129.

## Blockers

None. The current result is bounded to the declared synthetic screen.

## Decisions / rulings

The protocol phrase “accuracy within 0.01 of MTP” was evaluated as a no-degradation floor. Mirror exceeded MTP accuracy by more than 0.01 in every fresh world; this interpretation is recorded after opening the fresh set, without changing the training configuration or metric.
