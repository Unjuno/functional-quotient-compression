# MA-560 — Universally slimmable Mirror width code

Status: FAIL under the preregistered total-payload gate; scoped aligned synthetic quality signal. Frozen protocol: [`PROTOCOL.json`](PROTOCOL.json).

This is a small synthetic mechanism screen for width interpolation. One shared 16→32→16 linear network is sliced to nested hidden widths. The teacher rotates adjacent active channels by an angle affine in normalized width. A Mirror candidate stores that two-scalar angle law. Controls are prefix-only US-Net behavior, an affine direct sine/cosine coefficient law over the same shared maps, and a full independent width-map upper reference.

The study tests whether the added coordinate recovers width-conditioned functions on held-out widths. It does not test training a universally slimmable network, Transformer quality, or model capacity. The protocol charges actual NPZ bytes for shared maps and all method state; independent maps include one full operator per width.

## Development result (frozen before fresh)

Across worlds 56001–56002, held-out-width mean/max NRMSE was 0.2205/0.5250 for prefix-only US-Net, 6.9e-9/1.57e-8 for the affine-angle Mirror, 0.00661/0.02624 for the direct affine sine/cosine code, and 2.59e-8/2.81e-8 for independent full maps. Actual serialized payloads were 5,138B, 5,396B, 5,410B, and 9,032B respectively. Mirror interpolation passes development quality and beats the simple direct coefficient law on quality at almost equal bytes, but its total payload is 59.8% of independent maps and misses the frozen 50% byte gate. The exact development configuration is recorded in `DEV_FREEZE.json`; fresh is not used for tuning.

This is evidence on a deliberately Givens-aligned teacher only. It does not establish a trained slimmable network or general width-conditioned capacity.

## Fresh result and disposition

Across fresh worlds 56011–56013, held-out-width NRMSE mean/max was 0.2876/0.5404 for prefix US-Net, 7.51e-9/3.19e-8 for Mirror, 0.01123/0.03466 for direct affine sine/cosine coefficients, and 2.48e-8/2.84e-8 for independent full maps. Total serialized payloads were 5,138B, 5,396B, 5,410B, and 9,032B. Mirror saved 40.2% against independent maps, short of the preregistered 50% gate, and was only 14B smaller than the direct coefficient control. The registered candidate is therefore **FAIL overall**; the quality result is scoped to the deliberately Givens-aligned teacher.

Timing amendment A1 replays all five worlds with identical decode/setup and repeated-forward timing instrumentation. It changed no metrics, bytes, or payload hashes. Four verification tests pass; 160 held-out rows replay exactly. See `STATUS.md`, `AMENDMENT_A1.md`, and `VERIFICATION.json`.
