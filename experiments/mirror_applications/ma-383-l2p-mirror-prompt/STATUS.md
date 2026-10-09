# MA-383 status

- Status: **FAIL** on frozen development gates; fresh remained sealed.
- Branch: `research/ma-383-l2p-mirror-prompt-20261009`
- Protocol frozen before implementation/development: `18f5646f`
- Development worlds: 38301, 38302
- Fresh worlds 38311–38313: not opened
- Tests: 3 passed
- Stored inference payloads: 10; key identity, retrieval, bytes, hashes and output metrics replayed

## H / T / D / C / U

- **H:** Shared prompt basis + one Givens angle per prompt preserves retrieval/function quality at ≤60% actual pool bytes.
- **T:** Eight explicit or compressed prompts, paid eight-key pool, noisy task-identity-free queries, fixed prompt decoder; five methods; two development worlds.
- **D:** FAIL. Independent retrieval itself was 89.45%/89.70%, below 90%. Mirror used 66.8% of explicit bytes; oracle prompt NRMSE was .601/3.3e-5; retrieved NRMSE .6635/.4251.
- **C:** Retrieval noise invalidated all methods; seed 38301 also exposed unstable Mirror fitting. Generic basis coefficients matched full prompts.
- **U:** Natural L2P, continual task streams, task validity after development-selected retrieval calibration, fresh worlds.

See README for fact/interpretation/hypothesis labels and exact controls.
