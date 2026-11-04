# MA-674 status

- Status: FAIL (Mirror-specific claim; sparse quality gate)
- Branch: `research/ma-674-sparse-mirror-lora-atoms-20261104`
- Protocol frozen before development: yes
- Development: seeds 67401/67402
- Fresh/audit: sealed after exact native-control alias

## H — Hypothesis

Input-conditioned top-k selection over shared LoRA atoms can retain the full MoLE output with substantially lower complete bytes and active adapter work than full expert execution. Mirror must strictly improve over native sparse low-rank execution.

## T — Treatment and controls

Synthetic one-layer D=16, E=M=16, 512 inputs, random fixed router, aligned rank-1 atom experts. Compared full dense bank, top-2 full expert gate, top-k shared atoms for k=2/4/8/16, and native low-rank controls with identical factors and selection. Measured complete compressed NPZ bytes, NRMSE to full-bank output, MAC proxy, and CPU wall time.

## D — Decision

FAIL. At k=2/4/8, NRMSE across seeds was .46–.48/.35–.39/.19–.22, missing the .02 gate. k=16 was exact, with 6.21KB versus 17.8KB and 655,360 versus 2,359,296 MAC proxy. However native low-rank outputs and compute exactly matched Mirror; its payload was 2B larger because of field-name encoding, a negligible metadata artifact rather than a functional advantage. Full top-2 expert routing had NRMSE .85–.86. Fresh remained sealed.

## C — Strongest counter-hypothesis

The storage and compute gain comes from representing aligned dense expert matrices as ordinary shared rank-1 factors. Sparse atom truncation loses quality because many random coefficients contribute materially; no Mirror-specific freedom is demonstrated.

## U — Unconfirmed

No learned sparse router, natural trained adapter bank, accelerator latency, or task generalization is tested. MAC proxy does not establish wall-clock acceleration.

## Evidence labels

- Fact: full bank payload, output errors, and MAC proxies are recorded in `RESULTS_CORE.csv` and `source/development.json`; native atom controls match all Mirror outputs exactly.
- Interpretation: low-rank atom sharing compresses aligned expert banks; aggressive sparsity degrades quality.
- Hypothesis: structured or learned atom sparsity could help when expert coefficient distributions are naturally sparse, but must beat native low-rank routing.
