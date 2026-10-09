# MA-381 status

- Status: **FAIL** on frozen development payload gate; fresh remained sealed.
- Branch: `research/ma-381-lorahub-mirror-basis-20261009`
- Protocol frozen before implementation/development: `4a20efd1`
- Development worlds: 38101, 38102
- Fresh worlds 38111–38113: not opened
- Tests: 3 passed
- Stored inference payloads: 10; all bytes, hashes, source/few-shot test metrics replayed
- Independent LoRA gauge audit: non-orthogonal GL(2) transforms changed D, source outputs and composed predictions by <=3.6e-15

## H / T / D / C / U

- **H:** One source-specific SO(2) code over a shared rank-2 basis can preserve few-shot LoRAHub quality at <=60% independent-bank bytes and beat scalar gates.
- **T:** Eight source modules; four target tasks; 1,200 source updates and 300 signed-composition updates per target; seeds 38101/38102; independent/tied/scalar/generic-2x2/Mirror banks.
- **D:** FAIL. Mirror target-test NRMSE was essentially independent (.00000023/.00000018), but bytes were 60.9%/61.1% of independent versus the frozen <=60% gate. Scalar control quality was much worse; generic coefficients matched quality at ~4.1% more bytes.
- **C:** The source teacher is deliberately aligned to the tested rotation orbit; fixed overhead dominates the small eight-candidate total payload.
- **U:** Natural task adapters, task-disjoint generalization, larger banks and deployment latency.

See README for separate fact/interpretation/hypothesis labels and the aligned-feasibility boundary.
