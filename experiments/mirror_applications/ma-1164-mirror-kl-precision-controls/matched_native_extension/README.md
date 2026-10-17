# MA-1164 extension — param-matched native MLP controls

**State:** SCREENING COMPLETE — Mirror-specific PASS 0/3 worlds, total-byte PASS 0/3; result is limited to fixed Mirror geometry and sklearn digits.

**Mirror insertion:** a learned/native teacher width16 MLP (1210 logical weights) is represented by m paid physical coordinates plus fixed noncommuting K-head Mirror transforms interleaved with tanh and R residual recurrence. Runtime reconstructs exactly the same width16 NN topology before predictions.

A conventional narrow MLP with approximately the same trainable scalar count is trained from scratch; widths h6/7/10/11/13/16 correspond to 460/535/760/835/985/1210 scalars. FP16 and FP64 native inference payloads are measured. Same-m non-Mirror ortho generated controls remain mandatory.

**Critical qualification:** Mirror conversion inherits a previously trained width16 teacher. It is not learning from scratch with only m parameters and does not claim equal training compute. Costs for full teacher and compilation are reported.

New development partitions 618440/618441 choose only narrow-native LR. Documented fresh partition IDs 618450/451/452 are assessed with seeds51/52, no test tuning. A pre-existing source/protocol mismatch to 618460/461/462 was discovered; exploratory outcomes from that deviation are **not used** in the accepted results. See `FREEZE_CORRECTION.json` and `REPORT_JA.md` for full chronology.

To reproduce from the repository path `experiments/mirror_applications/ma-1164-mirror-kl-precision-controls/matched_native_extension/`:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q tests
# Use a clean copy with no results/fresh.csv to rerun:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python source/run.py --phase fresh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python source/run.py --phase audit
```

The runner imports `../source/core.py` and `../source/run.py`, already present on the MA-1164 branch. Do not edit the vendor code, parent negative reports, or `main`.
