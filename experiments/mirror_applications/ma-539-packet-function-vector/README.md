# MA-539 — Support-extracted function vector as a shared packet plan

Status: FAIL. Scope: synthetic function execution with four-slot packets.

## H — falsifiable hypothesis

A 16d behavior vector extracted from eight support pairs and broadcast to four slots will match an equal-width generic latent and PTP slot-code control on jointly correct packets while using at least 10% less serialized state.

## T — protocol and execution

PA99 motivated support-extracted behavior vectors; PA10 and the TM001/MA-248 evidence motivated packet-level consistency checks. Two development worlds each contain 16 independent random permutations over 32 states. Eight input-output pairs per function form support; the other 24 inputs form disjoint held-out queries grouped into 96 four-slot packets.

The shared width-32 MLP packet decoder was trained at all four frozen settings (800/1200 updates × LR .001/.003), using shared packet minibatches. The support-FV model averages an encoded representation of its eight support pairs. Controls were no function code, a directly learned generic latent bank, four independent PTP-style slot codes, and a serialized exact function table. Settings were selected by mean held-out token NLL of the FV candidate; 800 updates/LR .001 was selected. Fresh worlds 53911–53913 remain sealed because the absolute packet quality and byte gates fail.

All controls’ actual uncompressed NPZ payloads, world splits, mapping hashes, optimizer counts, matrix-operation proxy and wall times are retained under `results/`. Development and deterministic replay matched every metric and payload array exactly. Serialized payload byte sizes and hashes are in `payload_hashes.json`.

## D — FAIL

At the selected setting, the support FV had 0/96 exact packets in both worlds and 1.56–3.13% token accuracy. The generic latent and PTP slot-code controls also had 0/96 exact packets. The exact serialized table upper had 96/96 exact packets in 1,052 B. FV payload was 23,784 B versus 21,094 B for generic latent (1.127x; frozen maximum 0.90x). The FV token NLL was lower than generic latent by 0.353 and 1.583 nats, but this did not translate to any exactly correct packet.

## Fact / interpretation / hypothesis

- **Fact:** The neural methods had zero joint-packet accuracy in both worlds across the selected setting; the table upper was exact. FV payload exceeded generic latent payload by 2,690 B. The FV’s selected-setting token NLL was lower than generic latent in both worlds.
- **Interpretation:** Support-extracted function vectors did not establish a useful packet function representation or a storage win. The small NLL improvement is a fixed-budget signal on a task where all learned methods fail exact packet execution.
- **Strongest counter-hypothesis:** The random permutation task is information-limited: eight examples cannot identify the remaining 24 arbitrary mappings. The learned models’ near-zero performance says little about smoother/compositional functions.
- **U — limits:** Synthetic arbitrary permutations only; one packet width, one state count, one MLP scale; no natural text, compositional rule family, or optimized inference. Fresh worlds stayed sealed.

## Verification

Tests: `python -m pytest -q experiments/mirror_applications/ma-539-packet-function-vector/tests` (2 passed). Deterministic replay reproduced split arrays, all model arrays, selected and unselected metrics, and serialized sizes exactly. See `VERIFICATION.json`.
