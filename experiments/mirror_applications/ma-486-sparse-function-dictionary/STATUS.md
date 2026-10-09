# MA-486 status

- Status: **FAIL** for Mirror-specific attribution
- Branch: `research/ma-486-sparse-dictionary-functions-20261008`
- Base commit: `ab4acdf06f391bbad8b86bfbaeb762fb8bab6d22`
- Protocol frozen: yes (three amendments document a runner fix, heldout uniqueness accounting and compact uint8 IDs)
- Development complete: yes (48601, 48602)
- Fresh/audit opened: no (48611–48613 sealed)
- Verification: payload/hash replay and native OMP alias passed; tests 3 passed

## Next action

Proceed to MA-487 LISTA inference with matched dictionary, sparsity and OMP controls.

## Decisions / rulings

Sparse int8 reduced complete bytes by 11.8% versus dense int8 and decode operations by 90.6%, but missed the frozen <=20% byte reduction gate. Native OMP exactly aliases sparse Mirror codes.
