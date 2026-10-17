# MA-1163 — learned NN -> nonlinear recurrent Mirror weight program

**State:** SCREENING completed. Constructive exact representation PASS only at full independent coordinates; low-byte Mirror-specific advantage FAIL for the frozen direction family. Not an LLM/capacity proof.

**Mirror insertion:** replace independently stored MLP weights with a vector of `m` physical coefficients and fixed K=3 noncommuting Householder+Givens+permutation Mirror transforms interleaved with tanh and R=3 residual recurrences. The virtual 64→16→10 NN weights are synthesized from the resulting orbit.

## What this is and is not

1. A **normal** neural network is trained first on the true labels of the sklearn 8×8 handwriting digit dataset; all three initial seeds pass >90% dev accuracy.
2. Freeze the learned weights. Fit the recurrent Mirror representation **to the stored weights only** (not logits, labels, or student responses). Restore the virtual NN and execute it on previously heldout images.
3. Independently, train the same nonlinear Mirror program directly from image labels with AdamW to test whether representation and optimizer training behave differently.
4. A first-m fixed-direction experiment and a post-hoc dev-only top-m direction search are separated. The selected addresses are then tested on three **new** sealed split worlds, with indexes charged as stored bytes.

No one-scalar-only assumption: physical coordinate count `m=1,16,64,256,1210` in the main run. The separate selected-directions fresh replication uses m=512,768. A full m=1210 exact reparameterization is **not compression**.

## Direct controls and gate

- Ordinary NN with 1210 trainable FP64 weights: same input dimensions, tanh, hidden layer, logits, and topology.
- Same-m deterministic orthogonal seed expansion **without** the nonlinear Mirror recurrence.
- Fixed-first-m versus pretrained-weight-chosen top-|coordinate| Mirror addresses; selected indices stored in the payload.
- Native labels-only AdamW learning versus mirror labels-only AdamW learning. Mechanical reconstruction uses only weight vectors and LBFGS or a contraction inverse, not label-distillation.
- Mirror-specific win requires equal-m selected-Mirror to exceed plain selected orthogonal code by >=1 percentage point in >=2 of 3 new worlds; this gate FAILS.

## Run

From this directory with Python dependencies torch, numpy, scipy, scikit-learn, pytest, pandas:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m pytest -q tests
python source/run.py --phase baseline
python source/run.py --phase compile
python source/run.py --phase direct
python source/dev_direction_search.py
python source/replicate_direction.py
python source/audit.py
```

The scripts deliberately refuse to overwrite result CSV files. To rerun, use a clean copy of the project. See `REPORT_JA.md`, the three frozen protocols, `VERIFICATION.json`, and CSV under `results/`. Checkpoint `.pt` files are available in the conversation archive but not required for retraining reproducibility.

**Known limitations:** one small sklearn digits dataset, only 3 main seeds and 3×2 new split/seed replication, CPU FP64 only, Python eager Mirror implementation (no optimized fused kernels), naive fixed Mirror basis, byte accounting distinguishes payload versus separately charged execution-code source. No high-precision-bit or novel scalar optimizer claim.
