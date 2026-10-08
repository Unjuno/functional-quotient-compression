# MA-111 status

- Status: PROMISING; registered aligned fresh quality/storage gate passed 3/3.
- Branch: `research/ma-111-semantic-role-embedding-20261007`
- Base commit: `6e0ed10a094f9a082a144a2d8d86f508fd4cbad6`
- Protocol freeze: `03c101d1c9f37d0322283d053036a797766230f6`
- Fresh configuration freeze: `b9aca4126c348c02d16237c79f40f119fbac3998`
- Development: complete; LR .01 selected by the registered aggregate mean KL.
- Fresh: seeds 11111–11113 completed on aligned and independent teachers; all aligned quality/storage gates passed.
- Fresh split integrity: true.
- Results: 1,400 raw rows in `RESULTS_CORE.csv`.
- Verification: exact replay of 600 fresh metric/resource rows; 2 tests pass.

## H — hypothesis

A shared input embedding plus one Givens coordinate per semantic role preserves held-out-filler next-token distributions at lower actual bytes than private role maps; unrelated role transforms need private capacity.

## T — setup

Four known role IDs, 32 filler tokens, shared frozen 16D embeddings and 8-way readout. Compare hard tie, role addition, FiLM, rank-2 input/output LoRA, full per-role output heads, fixed VSA sign binding, full role maps, and Mirror Givens views. Development 11101/11102 selected LR .01; fresh 11111–11113 were opened only after both aligned development seeds passed.

## D — decision

PROMISING on the synthetic aligned role-view task: 3/3 fresh seeds passed absolute KL, top-1, calibration-delta, output-LoRA-margin, and actual full-map byte gates. Independent role maps required richer/private state. Runtime regressed against simpler adapters despite a lower MAC proxy.

## C — strongest counter-hypothesis

Rank-2 output LoRA implements the same role-conditioned linear change, matches the function at higher bytes, and runs faster. The exact-teacher packed-angle reference is also smaller than the current per-angle record format.

## U — unresolved

Natural semantic-role learning, causal-LM NLL, byte-matched generic embedding controls, near-convergence capacity, and an optimized packed-code runtime remain untested.
