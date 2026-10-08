# MA-452 — Factorized PathNet path × Mirror role address

Status: SCREENING  
Evidence lane: GENERALIZATION/STORAGE/RUNTIME  
Branch: `research/ma-452-factorized-path-view-20261008`  
Base commit: `085e918`  
Prior art: PA80 (PathNet), PA81 (Routing Networks)

## Hypothesis

H: Separate path identity from role address so a compact factorized Mirror codebook can define useful functions at path-role pairs never seen together during meta-training. Direct native conditioning and additive factorized controls test attribution.

**Mirror insertion:** this experiment attaches a learned two-angle role codebook independently of the selected PathNet path, allowing a shared module path and a role never jointly observed during training to compose into a new function.

## Frozen split and controls

There are four two-layer paths and three role addresses. The eight train pairs cover every path and every role, while four path-role pairs are withheld. Conditions are PathNet path-only, factorized Givens Mirror, rank-2 additive basis × role code, full role-vector table, native direct Givens conditioning and independent held-out task fits. The measured task is a small aligned 4D regression mechanism, not a full PathNet or Routing Network reproduction.

Two development worlds (45201, 45202) and the training/evaluation schedule are frozen in `PROTOCOL.json`. Fresh worlds 45211–45213 remain sealed unless all success gates pass. Every module, code, route/role ID, basis and schema byte is charged in actual uncompressed `.npz` inference payloads.

## H — Hypothesis

The frozen hypothesis and gates are in `PROTOCOL.json`: an independently trained role codebook should compose with paths on four unseen path-role pairs, reduce query RMSE versus PathNet path-only, and beat bytes/compute controls without a native alias.

## T — What ran

Two development worlds (45201, 45202), each with 160 outer updates × 8 training episodes of 48 examples (61,440 examples), trained on eight path-role pairs. Four unseen pairs cover all path and role identities. Held-out quality is averaged over 48 query examples per pair. Controls: PathNet path-only, two-angle factorized Mirror, rank-2 factorization, full role-vector table, direct native Givens conditioner and independent support-fit task vectors. Fresh worlds 45211–45213 remain sealed. `RESULTS_CORE.csv` records all per-condition updates, compute proxies, wall time, actual payload bytes and hashes.

## D — Result: FAIL

| Method | 45201 RMSE / bytes | 45202 RMSE / bytes |
|---|---:|---:|
| PathNet path-only | 1.0623 / 1,468 | 1.2528 / 1,468 |
| Factorized Mirror | 0.2532 / 1,766 | 0.2680 / 1,766 |
| Native Givens | 0.2532 / 1,766 | 0.2680 / 1,766 |
| Rank-2 basis × role | 1.0441 / 2,032 | 1.1388 / 2,032 |
| Full additive role vectors | 0.9287 / 1,772 | 0.9789 / 1,772 |
| Independent task vectors | 0.0226 / 1,234 | 0.0318 / 1,234 |

**Fact:** The factorized model generalizes to all four unseen path-role pairs in both worlds. Its RMSE is 0.24×/0.21× path-only, 0.27×/0.27× the full additive role table, and it uses 13% fewer bytes than the rank-2 control. Yet its actual payload is 20% larger than path-only and 43% larger than independently fitted task vectors. The native Givens control matches the 1,766-byte payload hash and output RMSE exactly in both seeds. Independent task vectors are much more accurate. Training took 1.29/1.03 s for Mirror versus 0.52/0.47 s for rank-2. Four-task query timings were below 0.11 ms; seed 45202 missed the 1.25× rank-2 runtime threshold, but this microbenchmark is too short to support a reliable runtime conclusion. Payload replay and serialization verification pass; fresh worlds remain sealed.

**Interpretation:** Factorized role coordinates transfer across path-role combinations absent from training, which is a useful aligned compositional result. However, the implementation is precisely an ordinary role-addressed Givens conditioner, so no Mirror-specific value is established. It also does not improve the storage frontier against path-only or independent task vectors.

**Hypothesis:** The factorization works when task structure really is path × role, but simple role conditioning can express the same rule; this screen does not show natural program transfer or a favorable quality/byte Pareto point.

### Worker report

- **H:** A separate path factor and role codebook can form useful functions at combinations never seen together during training.
- **T:** Eight train pairs and four disjoint held-out pairs across two development seeds; path-only, rank-2, additive role, native Givens and independent task-vector controls; actual payload/output replay passed.
- **D:** **FAIL** (held-out compositional quality is real, but native Givens exactly aliases it and bytes miss both path-only and independent-fit gates).
- **C:** Standard role-addressed Givens conditioning explains the full effect; independent fit remains substantially stronger.
- **U:** Natural task pathways, larger unseen-combination banks, learned routers and optimized kernels.

## Verification

`source/verify.py` reopens every payload, checks its length/hash, replays every held-out query metric from serialized state, confirms exact Mirror/native equality, and confirms that fresh seeds stayed sealed. Four focused tests passed.
