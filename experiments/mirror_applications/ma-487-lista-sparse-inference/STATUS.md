# MA-487 status

- Status: **FAIL**
- Branch: `research/ma-487-lista-sparse-function-inference-20261008`
- Base commit: `7f3f6586836fe3e34df6cb45ea1430871484a941`
- Protocol frozen: yes
- Development complete: yes (48701, 48702)
- Fresh/audit opened: no (48711–48713 sealed)
- Verification: serialized replay passed; tests 3 passed

## Next action

Proceed to MA-488 shared/private dictionary and Mirror coefficients; require a heldout heterogeneity sweep and an explicit private-atom control.

## Decisions / rulings

LISTA passes quality/sparsity and beats OMP compute, but direct projection plus top-3 has better quality with similar/lower operations and bytes. FAIL; fresh sealed.
