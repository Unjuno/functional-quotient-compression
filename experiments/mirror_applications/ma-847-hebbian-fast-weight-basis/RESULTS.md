# MA-847: NOT ESTABLISHED

## H / T / D / C / U

- H: shared Hebbian fast-weight basis plus transient coefficients can preserve one-shot recall with lower writable-state cost.
- T: synthetic D=16 associative memory; attempted 8 support pairs and 64 queries; dev seed 13, fresh 31/53. Compared full outer-product state, native coefficients, Mirror code, rank2 direct factorization. Fresh outputs were inspected after protocol/code freeze, but the screen had a validity flaw.
- D: NOT ESTABLISHED. Full Hebbian update did not span the query space (nRMSE .734–.803). Coefficient and Mirror modes derive their code from the teacher matrix rather than support examples (nRMSE .074–.078), so their apparent quality is oracle leakage. Fresh metrics are preserved as invalid exploratory rows; no retuning after fresh.
- C: incomplete support coverage and teacher-oracle code construction invalidate the comparison.
- U: all useful recall, byte-frontier, write-cost, and Mirror-specific claims; trained fast-weight learning remains untested.

## Interpretation

The measured rows are retained for audit and must not be treated as evidence. A valid successor must encode queries from the actual support outer products, use support keys spanning the evaluated query domain or explicitly score only retrieved keys, and count the basis persistently once plus transient code per episode. Since fresh seeds were opened, any corrected protocol requires an amendment/new experiment and new fresh seeds.
