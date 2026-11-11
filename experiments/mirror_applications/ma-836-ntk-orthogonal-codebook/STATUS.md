# MA-836 status

Status: **FAIL** — scoped synthetic NTK geometry/storage screen. Fresh data remained sealed after the rank-2 quality miss and exact native factorization alias.

## H / T / D / C / U

- **H:** NTK-weighted task coding would retain heldout quality with at least 25% fewer bytes than independent updates, reduce inter-task overlap versus Euclidean rank-2, and beat native weighted factorization.
- **T:** Development seed 83601; eight 32D tasks; 96 support and 256 heldout examples/task; closed-form readouts, rank-2 factorization, no optimizer updates. After amendment 01, also measured full-rank Euclidean and NTK-orthogonal bases. Focused tests: 1 passed.
- **D:** FAIL. Independent readouts reached nRMSE 2.08e-15 at 1,614 B. Rank-2 Mirror/native NTK factorization exactly aliased at 1,117 B but nRMSE was .09086, above .02. It also had slightly higher inter-task NTK overlap than Euclidean rank-2 (.63159 vs .62830). Full-rank NTK-orthogonal basis attained 3.27e-15 error and 2.2e-15 basis orthogonality error, but cost 2,065 B and a 9x forward MAC proxy; its task-function overlap equaled independent readouts (.60188). Euclidean full-rank orthogonal basis also exceeded independent bytes (2,055 B). Fresh seeds remained sealed.
- **C:** Rank-2 compression is incompatible with preserving this bank's eight independent functions at the chosen tolerance. An orthogonal basis can be formed at full rank, but it does not reduce task-function interference and costs more bytes/compute. The NTK metric weighted-SVD path is ordinary native factorization.
- **U:** No trained network, actual PEFT/NTK-CL, continual learning sequence, unseen-task generalization, or fresh-seed evidence. The teacher is synthetic and task readouts are stored.

## Fact / Interpretation / Hypothesis

- **Fact:** Mirror/native rank-2 payload SHA-256s, bytes and outputs match. Rank-2 misses quality. Full-rank NTK basis is orthogonal and exact but larger than independent storage; function overlap is unchanged from independent updates.
- **Interpretation:** Orthogonality of a shared basis does not by itself reduce interference between the represented task functions. The required rank erases storage savings in this screen.
- **Hypothesis:** NTK geometry may help allocate low-rank PEFT codes only when task functions are already compressible at the required quality; a codebook orthogonality constraint alone is not evidence of less functional interference.
