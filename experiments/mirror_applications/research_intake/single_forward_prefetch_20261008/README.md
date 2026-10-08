# Single-Forward Multi-Mirror + Group Prefetch (SFM001/002)

**Independent research only / not the active worker queue.** The goal is to test whether one expensive shared execution can supply four/five functional Mirror outputs while the next physical expert group is prefetched. This is a physical compute/memory/pipelining hypothesis, NOT a claim that rotating an arbitrary hidden state after a Transformer forward reproduces a new trained model.

## Research navigation

- [Pre-frozen main experiment](FROZEN_PROTOCOL.json) — seed firewall, exact/approximate gates, timings and hypothesis.
- [Separately frozen folded-expert control](FROZEN_FOLDED_CONTROL_AMENDMENT.json) — native ordinary folded weight speed versus extra resident bytes.
- [Stage-0 report and exact proof](REPORT.md) — algebra PASS where factorization exists; hidden nonlinear output-only generality FAIL; K4/K5 timing results; hypothetical transfer simulator; full provenance.
- Full standalone [MA-1175 plan](plans/MA-1175/README.md) and [MA-1176 plan](plans/MA-1176/README.md).
- [MA-231/MA-691/MA-253 cross-over safeguards](existing/README.md) avoid duplicating historical folded-weight, KV, and final-FFN placement evidence.
- [Reproducibility and original 10+2 unit tests are in the [experiment ZIP/runbook](REPRODUCE.md); the repository has a directly runnable [algebra preflight](pilots/sfm001/source/algebra_reference.py).

The authoritative live-worker branch remains `research/mirror-application-worker-ready-20261007` with its pre-existing next candidate MA-255. Its queue, claims and main were not modified.

## Run conditions and scientific firewall

All tests are **isolated** on `research/mirror-single-forward-prefetch-20261008`, dev seeds 11–13, fresh 101–105. A separately preregistered folded baseline used dev 21–23 and fresh 201–205. Only a one-thread AMD EPYC CPU PyTorch 2.10 eager implementation and a virtual two-buffer transfer simulator are available; there is no CUDA here. Warmup 20, measured iterations 100 per setting; model dimensions and branches are fully in [frozen SFM001 protocol](FROZEN_PROTOCOL.json) and [folded amendment](FROZEN_FOLDED_CONTROL_AMENDMENT.json). No result is a real PCIe transfer, full LLM, or independently trained MoE deployment.

**Training audit:** With a shared input projection z, the per-View losses' gradients w.r.t. z add; exact backprop does not require all Views to be parallel (sequential accumulation is valid). Parallelization is optional hardware scheduling, not a mathematical prerequisite.

**Storage:** Actual NPZ for shared/tied and prefolded resident matrices; count generator, View codes, per-role trained readout, metadata and buffers in further lanes. A k-fold expansion is NOT k independent trained experts.

**Required evidence:** `RESULTS_CORE.csv`, dev/fresh CSVs, `VERIFICATION.json`, source/test hashes, hardware and launch provenance, downstream task utility, task-identity holdout and native competitors before status changes.
