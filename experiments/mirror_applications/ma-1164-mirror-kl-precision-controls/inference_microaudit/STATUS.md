# MA-1164 on-demand Mirror microaudit — 2026-10-18

- **POSTHOC DIAGNOSTIC** separate from the preregistered MA-1164 image task.
- Packed inference state test PASS; compact Mirror FP16 payload 1710 B vs compact dense FP16 2442 B on synthetic coefficients; executable source not included in either payload.
- Matrix-free structured Mirror operator PASS for 6 (d,K,R) combinations; numerical error <1e-11.
- Nonlinear Mirror sandwich functional and scalar-vector gradient tests PASS, 7/7 tests overall.
- Eager CPU direct structured Mirror slower than cached dense matrix, even though it uses far fewer weight bytes.
- Existing MA-1164 classifier remains not Mirror-specific; no previous M0/FAIL verdict changed.
- Previously opened MA-1163 world 618201 and 3 saved native seeds: compact Mirror FP16 96.389% / 1710 B vs compact native FP16 97.407% / 2442 B; diagnostic only, not fresh.
