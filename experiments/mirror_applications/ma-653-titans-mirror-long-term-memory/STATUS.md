# MA-653 status

- Status: FAIL
- Branch: `research/ma-653-titans-mirror-long-term-memory-20261104`
- Base commit: `9bc5fb90`
- Protocol frozen before development: yes
- Development complete: yes (65301, 65302)
- Fresh/audit opened: no; exact native low-rank alias gate reached

## H / T / D / C / U

**H:** A shared value basis plus per-key codes compresses surprise-gated long-term memory; private residual rank identifies when key-specific value directions are needed. Short-term cache state is separately charged.

**T:** Titans-inspired online fast-weight store: 64 one-hot keys, 32D values, four shared directions, private residual ranks 0/2/4/8, two writes/key, threshold .12, and an exact 8-entry cache of the newest distinct keys. Two development seeds. Compared full writable matrix, shared rank4, shared+private factor, identical native low-rank factor and explicit KV table. Serialized complete, long-term and cache payloads; measured cache-aware/long-term query MSE, Gaussian NLL, surprise writes and compute proxy. Fresh seeds 65311–65313 remained sealed.

**D:** FAIL for Mirror-specific claim and frozen complete-byte gate. Shared-only rank4 at residual rank0 reaches effectively zero long-term MSE with 43.2–43.3% of full complete bytes (full is ~10.4KB); the 8-entry cache adds the same state to all methods. At private rank2/4, shared+private restores near-zero long-term error with 54.8–62.0% of full bytes; rank8 needs 75.6–75.7% and misses the <=50% gate. Shared-only MSE rises to .0011/.0023/.0050 at target private rank 2/4/8; its surprise gate triggers 106/125/128 writes versus 64 for full and matched private-rank memory. Mirror and native low-rank state, output and bytes are exactly equal. Fresh stayed sealed.

**C:** All compression comes from ordinary low-rank factorization of key-value memory; the Mirror coordinate is a standard per-key basis code. The 50% total-state gate is reachable only when values lie almost entirely in the shared rank4 span.

**U:** This is a linear Titans-inspired associative memory, not the neural long-term memory architecture or paper benchmark. One-hot keys remove retrieval competition; short-term cache is exact. Natural sequence surprise, semantic retrieval and language-model quality are untested.

## Fact / Interpretation / Hypothesis

- **Fact:** 40 development rows; all complete/long-term/cache actual bytes and query metrics replay; three tests pass; protocol hash matches freeze.
- **Interpretation:** The low-rank frontier is strong for aligned long-term values; private residual rank restores diverse values but pushes complete storage beyond the stringent 50% target. Surprise gating repeats writes when the model lacks private directions.
- **Hypothesis:** Larger memory banks with correlated value directions may amortize bases, but Mirror must beat a matched native low-rank memory control to establish distinct value.
