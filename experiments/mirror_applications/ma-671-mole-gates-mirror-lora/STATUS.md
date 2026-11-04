# MA-671 status

- Status: FAIL (Mirror-specific claim)
- Branch: `research/ma-671-mole-gates-mirror-lora-20261104`
- Base commit: `57147cde`
- Protocol frozen before development: yes; development-only amendment recorded in `PROTOCOL.json` and `freeze.json`
- Development complete: yes (seeds 67101, 67102; three residual magnitudes)
- Fresh/audit opened: no; stopped at the prespecified native-control alias gate

## H — Hypothesis

Layerwise input-dependent MoLE gates over a small shared LoRA atom bank can reproduce the full expert bank at materially lower serialized inference bytes and active MACs. A Mirror-specific result additionally requires a strict gain over a native shared-basis MoLE control.

## T — Treatment and controls

Synthetic two-layer, width-16 MoLE; 8 experts per layer; rank-4 shared atom bank; 256 inputs; residual magnitudes 0, 0.05, and 0.2; two development seeds. Controls: dense full MoLE, full MoLE with top-2 routing, native shared-basis MoLE, and matching rank-2 private residual variants. Complete inference payload sizes are measured from compressed NPZ files. Fresh seeds remain sealed because Mirror and native shared-basis controls alias exactly.

## D — Decision

FAIL for Mirror-specific advantage. In both seeds and all conditions, Mirror and native shared-basis output arrays are exactly equal. The native payload is 2 bytes larger in the tested encoding. At rho=0, shared mode uses about 5.97 KB versus 15.8 KB for full MoLE (about 38%); MAC proxy is 737,280 versus 1,245,184 (about 59%), missing the preregistered 50% compute gate. At rho=.05/.2, shared-only NRMSE is about .022/.09; private rank-2 residual restores zero measured error at about 14.1 KB and 999,424 MACs. Full top-2 has lower MAC proxy (458,752) but materially degraded output (NRMSE about .43–.68).

## C — Strongest counter-hypothesis

The effect is ordinary shared low-rank factorization and gate linearity; the native shared-basis control reproduces the function exactly, with a slightly smaller payload.

## U — Unconfirmed

No fresh-world generalization, learned gate optimization, benchmark-scale MoLE quality, or hardware latency result is established. This synthetic screen does not establish model capacity.

## Evidence labels

- Fact: the exact serialized NPZ payload sizes and output errors are recorded in `RESULTS_CORE.csv` and `source/development.json`; the runner asserts exact Mirror/native outputs and coordinate arrays.
- Interpretation: shared atoms can represent aligned logical expert banks compactly, while private residuals buy off-basis quality at additional storage and compute.
- Hypothesis: meaningful Mirror-specific MoLE gains would require a view transform not reducible to native shared-basis factorization.
