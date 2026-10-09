# MA-482 status

- Status: PROMISING (scoped synthetic rate-distortion/storage result)
- Branch: `research/ma-482-residual-vq-mirror-20261009`
- Base commit: `69027d28`
- Protocol frozen: `3d711bfe`; A1 amendment documented before A1 fresh rerun
- Development complete: yes
- Fresh/audit opened: yes (A1 worlds 48220-48222; A0 excluded)
- Results committed: yes (`cac6f3a8`)
- Verification committed: yes
- Registry row updated: yes

## Decision

H: A shared known residual plane with compact phase indices can reproduce native residual-VQ quality at substantially fewer actual bytes.

T: Synthetic 16D bank, 128 functions, R={1,2,4}, K={4,8,16,32}; 3 A1 fresh worlds × 3 seeds; dense, learned native RVQ, generic explicit coefficient table, and deterministic Mirror phase-grid decoder. A1 infers phase from target vectors with `atan2`; A0's oracle phase results are preserved as exploratory and excluded.

D: PROMISING only for this known orthogonal product-of-circles family. At R4/K32, Mirror is 3,037B and mean normalized error .0578; native RVQ is 10,597B and .1328; generic coefficient control is 3,545B and the exact same reconstruction error. The Mirror payload is 28.7% of native and 85.6% of generic control. Deeper stages reduce error on held-out targets, but they also grow payload. This is a strong structured-code storage result, not evidence of Mirror-specific quality or general capacity.

C: A hand-specified orthogonal basis, scales and phase grid encode the teacher's structure. Generic explicit coefficient tables recover identical functions; learned RVQ is weak under this finite development budget.

U: Learned/non-orthogonal functions, learned phase inference, optimized equal-byte frontiers, broader natural workloads, and efficient batched deployment kernels remain untested.

## Next action

Run registry-integrity check, artifact replay tests, update registry/claim ledger/board, then continue to the next untested P0.

## Blockers

None.

## Decisions / rulings

- A0 fresh results used oracle teacher phases and are retained in `artifacts/a0_*_exploratory.*`; they are excluded from confirmatory evidence.
- The first A1 runner attempt wrote only the final loop method to RESULTS_CORE.csv. It is not used; the corrected runner emitted all 432 control rows before metrics were inspected.
