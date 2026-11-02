# MA-560 — Universally slimmable Mirror width code

Status: SCREENING. Frozen protocol: [`PROTOCOL.json`](PROTOCOL.json).

This is a small synthetic mechanism screen for width interpolation. One shared 16→32→16 linear network is sliced to nested hidden widths. The teacher rotates adjacent active channels by an angle affine in normalized width. A Mirror candidate stores that two-scalar angle law. Controls are prefix-only US-Net behavior, an affine direct sine/cosine coefficient law over the same shared maps, and a full independent width-map upper reference.

The study tests whether the added coordinate recovers width-conditioned functions on held-out widths. It does not test training a universally slimmable network, Transformer quality, or model capacity. The protocol charges actual NPZ bytes for shared maps and all method state; independent maps include one full operator per width.

## Development result (frozen before fresh)

Across worlds 56001–56002, held-out-width mean/max NRMSE was 0.2205/0.5250 for prefix-only US-Net, 6.9e-9/1.57e-8 for the affine-angle Mirror, 0.00661/0.02624 for the direct affine sine/cosine code, and 2.59e-8/2.81e-8 for independent full maps. Actual serialized payloads were 5,138B, 5,396B, 5,410B, and 9,032B respectively. Mirror interpolation passes development quality and beats the simple direct coefficient law on quality at almost equal bytes, but its total payload is 59.8% of independent maps and misses the frozen 50% byte gate. The exact development configuration is recorded in `DEV_FREEZE.json`; fresh is not used for tuning.

This is evidence on a deliberately Givens-aligned teacher only. It does not establish a trained slimmable network or general width-conditioned capacity.
