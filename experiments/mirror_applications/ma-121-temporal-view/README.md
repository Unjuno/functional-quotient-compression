# MA-121 — packet phase-slot Mirror head

Status: PROMISING (synthetic aligned mechanism)
Evidence lane: MECHANISM
Base commit: `e04e131926978fada98129d52e2146534a8db6e9`

## Hypothesis

H: One shared output head plus per-slot Givens coordinates can recover four future-token slot functions with lower payload than independent MTP heads while retaining exact packet consistency and beating a low-rank slot-code control on the storage/quality frontier.

## Physical-to-logical claim

- Physical object: one 8-input, 8-token output head.
- Mirror coordinate: one learned Givens angle per future slot.
- Logical multiplicity: four slot-specific token functions.
- Failure modes: generic rank-2 slot bases can fit the teacher; independent slot functions may require MTP heads; eager Mirror decoding may be slower.

## Prior-art delta

PA09 establishes multi-token heads from a shared trunk. PA10 conditions joint prediction on sequence randomness. This screen compares independent MTP, a rank-2 shared slot-code control, and a small token-feedback decoder. The latter does not implement Transformer KV caching; no full cached-AR claim is made.

## Fresh results

The synthetic aligned teacher applies four Givens views to one shared output matrix. The independent condition uses unrelated slot matrices. Each world provides 256 training examples and 128 test examples. All learned methods train 500 Adam updates at LR 0.02.

| Method | Median aligned test NLL | Exact four-token accuracy | Payload | Inference tokens/s |
|---|---:|---:|---:|---:|
| tied shared head | 0.4005 | 0.641 | 1,961B | 19.2M |
| rank-2 slot-code control | 0.1750 | 0.820 | 2,849B | 9.68M |
| Mirror phase-slot head | 0.2474 | 0.828 | 2,149B | 3.13M |
| independent MTP heads | 0.3032 | 0.664 | 2,729B | 26.3M |
| sequential token-feedback decoder | 0.7282 | 0.648 | 2,721B | 4.49M |

Across all three fresh aligned worlds, Mirror improved NLL and exact packet accuracy over MTP while using 21.3% fewer payload bytes. It used 24.6% fewer bytes than the rank-2 slot-code control, while that control achieved lower NLL; these are distinct storage/quality points. In the independent-slot condition, Mirror exact packet accuracy was 0–2.3%, versus 53.9–67.2% for MTP, showing the need for private slot capacity.

The eager CPU Mirror path generated 3.13M tokens/s, 0.119x the MTP head throughput. Its MAC proxy was 9.4% above a shared linear projection, so the larger measured slowdown reflects unoptimized view construction/Python overhead in this tiny screen.

## Decision

**FACT:** Aligned fresh gates passed in 3/3 worlds: NLL was at most 1.10× MTP, packet accuracy was no worse than MTP, and payload was 2,149B vs 2,729B. Mirror accuracy was better than MTP by more than 0.01 absolute in these worlds; the protocol's “within 0.01” phrase is interpreted as a no-degradation floor. PTP rank-2 produced lower NLL at larger bytes. Fifty rows replayed with exact payload bytes and maximum metric delta 4.85e-10; tests passed 2/2.

**INTERPRETATION:** A phase-slot coordinate yields a useful storage/quality point for a teacher that follows the same phase family. It does not improve runtime, and independent functions still need independent heads.

**HYPOTHESIS:** Phase views may help when future-slot functions lie on a compact orbit; packet consistency should be tested on natural token sequences and exact decoding paths.

**BOUNDARY:** Synthetic frozen-feature classification only. No language-model NLL, autoregressive cache, MTP training loss, speculative acceptance, or end-to-end token latency claim.
