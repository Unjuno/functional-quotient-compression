# MA-258 — PSP and Mirror unbinding for logical expert banks

Status: SCREENING
Evidence lane: MECHANISM / STORAGE / RUNTIME
Base commit: `54abf8c2f768e20f071c3e62d6b5a5108bebc313`
Prior art: PA16 Parameter Superposition; PA01 expert tying across depth

## H — falsifiable hypothesis

For a small bank of related linear experts, expert-addressed orthogonal Mirror unbinding can preserve each expert's function using less serialized inference state than independent experts and less interference than native parameter superposition (PSP), while improving the byte/quality frontier over hard tying and a shared low-rank coefficient control. Unrelated expert maps should reveal the point where private parameters become necessary.

> **Mirror insertion:** this experiment adds a persistent per-expert orthogonal coordinate `m_e` at the output-channel interface of one shared physical expert matrix, so related logical expert maps can be recovered without storing one full matrix per expert.

- Native method before `m`: a tied expert or a PA16-style superposed expert tensor decoded by its task context.
- Physical object: one square linear expert matrix; each logical expert receives an explicit expert ID (oracle routing).
- Mirror coordinate: a per-expert angle that rotates two output channels of the shared matrix.
- Persistent `m`: one FP16 angle per expert; this is paid inference state.
- Simpler controls: hard tying; a shared low-rank basis with per-expert coefficients; native binary PSP with reproducible random sign contexts.
- Closest prior evidence: MA-255 tested task-map parameter superposition, and MA-257 found a native PA16 rotational context exactly matched the tested compositional Mirror context. MA-258 changes the unit to an expert bank with explicit expert-address retrieval and an unrelated-expert stress condition.

## T — frozen mechanism screen

For each seed, construct 8 expert matrices of shape 16×16. The aligned bank is generated from one shared matrix by a known, expert-specific output-channel Givens rotation. The unrelated bank consists of independent Gaussian matrices. Evaluate squared function error on 256 held-out Gaussian inputs per expert; task/expert identity is provided to every method, so router learning, routing errors and router storage are outside this screen. There are zero optimizer updates: this is a post-fit representation screen, not a training or capacity result.

Compare: independent full experts; hard tying by the mean matrix; rank-r shared SVD basis with expert coefficients (r ∈ {1,2,4,8}, serialized directly); PA16 binary sign-context PSP over a superposed tensor; and the one-angle Givens Mirror view. PSP sign matrices are generated from one recorded seed and charged as packed bits plus seed/shape metadata. All methods serialize to deterministic NPZ and are scored from the reloaded payload. The Givens angle construction is known to the view method and its per-expert angles are charged; this is an aligned, oracle-address upper screen for the Mirror view.

Development seeds: 25801, 25802, 25803. Select SVD rank by mean aligned normalized MSE under the byte gate; all other configurations are fixed. Fresh seeds locked but unopened: 25811, 25812, 25813. Fresh data will only be run if a development result meets the promising gate. The unrelated bank is a predeclared private-state stress test, not a tuning target.

## Gates

**PROMISING / proceed to fresh:** on all three development seeds, aligned Mirror normalized MSE ≤1e-6, actual payload ≤50% of independent, and either (a) at least 10% fewer bytes than the best non-Mirror method at MSE ≤1e-6 or (b) at least 10× lower MSE than the best non-Mirror method at no more than 10% extra bytes. PSP is the required direct native comparator.
**FAIL:** Mirror misses the aligned quality gate, or it fails both incremental gates against PSP and the shared low-rank control, or unrelated maps require essentially independent-sized private state without useful recovery.
**NOT ESTABLISHED:** serialization/reload mismatch, invalid deterministic replay, or a protocol execution defect.

These are mechanism-screen gates only; passing them would not establish learned routing, trained MoE behavior, natural-language quality, near-converged capacity, or an end-to-end MoE Pareto improvement.

## Storage and compute contract

Actual serialized inference payload bytes from reloaded NPZ are authoritative. Charge every matrix, angle, expert coefficient, packed PSP context, seed/shape metadata, and archive header. Report parameter bytes as a diagnostic only. Report 256 examples × 8 experts per world, zero optimizer updates, matrix-multiply MAC proxy, encode/decode/inference wall time, and throughput. Expert ID is an oracle supplied input and is stated as such; router cost and quality are not measured. CPU only; no nanoGPT/vendor edits.

## C / U

**Strongest counter-hypothesis:** the Givens teacher is constructed from the same one-angle chart, while exact expert addresses are supplied. Even a positive would show aligned representation compression, not a general benefit over trained expert tying or router-conditioned MoE. Native PSP and shared-basis controls may erase any Mirror-specific margin.

**Unconfirmed:** learned expert routing, noisy/load-balanced MoE behavior, optimized inference kernels, training from examples, natural language, arbitrary expert families, and fixed-byte near-convergence.

## Decision

### H / T / D / C / U

**H:** expert-addressed orthogonal views should recover an aligned logical expert bank at fewer bytes than independent experts, PSP and a simple shared basis; arbitrary maps should expose the private-state boundary.

**T:** 8×16×16 linear expert bank; 2,048 held-out input/expert examples per world; 0 optimizer updates; 3 development seeds (25801–25803) selected SVD rank 2; fresh seeds 25811–25813 evaluated that locked rank. Comparators were full independent, hard-tied mean, rank-2 shared SVD, binary-context PSP and Mirror. Expert IDs were supplied to all methods; there is no learned router. Actual deterministic NPZ payloads were reloaded before scoring. Six worlds × two conditions × five methods = 60 result rows. Four tests passed; metric/byte/hash replay had zero differences across all 60 rows.

**D: PROMISING, narrowly scoped aligned representation screen.**

**Facts:** Fresh aligned worlds: Mirror normalized MSE was 8.99e-10–1.07e-9 at **1,514B**; rank-2 SVD was 9.01e-15–9.20e-15 at **3,822B**; independent was numerical zero at **8,440B**; PSP was 4.62–5.46 at **1,766B**; hard tying was 0.0171–0.0202 at **1,270B**. Thus Mirror passed the frozen <=1e-6 quality gate and used 60.4% fewer payload bytes than the best simple control that also passed that quality gate (rank-2 SVD), and 82.1% fewer than independent. PSP used 14.3% more bytes than Mirror and had substantially higher interference error. Development results selected rank 2 under the preregistered byte-constrained rule.

Fresh unrelated worlds: Mirror normalized MSE was 0.848–0.889 at 1,514B, essentially the same error as hard tying (0.855–0.888 at 1,270B); rank-2 SVD improved this only to 0.565–0.597 at 3,822B. Independent maps remained near zero at 8,440B. Cheap views did not recover unrelated experts: reaching independent quality required private/richer state in this tested family.

The arithmetic proxy was 288 operations/example for Mirror, 256 for independent/tied, 512 for PSP and 1,280 for rank-2 SVD under the frozen conservative per-query materialization accounting. Fresh batched NumPy throughput averaged 11.16M examples/s for Mirror, 17.88M independent, 9.90M tied, 14.17M SVD and 3.93M PSP on aligned tasks. This small CPU harness does **not** show a runtime win: Mirror was slower than independent and timing depends on batching/materialization. Mean single-call encoder times are sub-millisecond and are only rough calibration. All methods saw 2,048 held-out examples/world and zero training examples/updates; one exact expert address was supplied per example.

**Interpretation:** A one-angle orthogonal chart can be a more byte-efficient code than an unstructured rank-2 coefficient representation when expert matrices lie on that same known orbit. Standard random-sign PSP had strong cross-expert interference here. Hard tying remains the smallest and fast option when a 1.9% normalized error is acceptable. This result is not a general superposition or MoE finding.

**C — strongest counter-hypothesis:** the target matrices were generated by the same two-channel Givens family used by Mirror; exact expert IDs and angles were oracle-known. The result is an aligned post-fit codec comparison. Better structured non-Mirror controls, different expert/task distributions, or trained expert/router systems may close the byte gap. The simple rank-2 SVD already matches quality, just with more stored basis state.

**U:** trained MoE behavior, router and load-balance cost, deep/nonlinear expert functions, arbitrary task-bank composition, optimized kernels, near-converged fixed-byte capacity, and natural-language quality remain untested. No logical-address count is interpreted as independent capacity.

Evidence: `RESULTS_CORE.csv`, `FRESH_RESULTS.csv`, `source/dev_selection.json`, `source/replay_verification.json`, and `VERIFICATION.json`.
