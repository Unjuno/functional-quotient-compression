# MA-752 status

- Status: FAIL (scoped synthetic Mirror-specific claim)
- Branch: `research/ma-752-controlnet-depth-views-20261109`
- Base commit: `b9c89c34`
- Development: complete, seed 75201; five controls.
- Fresh/audit: unopened after exact native factorization alias and byte miss.
- Verification: pending.

## H / T / D / C / U

- **H:** A condition × injection-depth coordinate factorization saves >=25% complete bytes while predicting held-out operator combinations.
- **T:** 4×4 synthetic operator family, 4 held-out pairs, 12×12 operators, independent, shared, additive, native factorized and Mirror controls.
- **D:** **FAIL.** 6,989 B vs 8,962 B is 22.0% savings; exact native alias.
- **C:** The generated operator family is factorized by construction, so native factorization is sufficient.
- **U:** Trained adapters, real diffusion controls and image metrics.

## Next action

Run the three focused tests, deterministic replay and registry integrity check, then commit and push this scoped FAIL.

## Blockers

None.
