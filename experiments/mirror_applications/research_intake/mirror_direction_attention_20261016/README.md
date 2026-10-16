# Mirror Direction Attention — Exploratory CPU Pilot (2026-10-16)

**Status: IDEA INTAKE / MECHANISM CHECK ONLY. No model training, no next-token loss test, no new MA ID, and no capacity or Mirror-specific speed claim.**

The model is a single causal token-attention computation plus low-rank cross-Mirror attention at the same token index. It compares that factorization to one ordinary token attention, four separate token-attention paths, and full attention over the token × Mirror grid.

**Results:** With batch=2, d=64, r=16, K=4, sequence length T=512, the median CPU times were 1.9240ms (one token path), 7.0998ms (four separate token paths), 2.5099ms (shared token + view attention), and 34.6184ms (full token × view attention). The factorized approach is ~2.83x faster than four separate paths **but 1.30x slower than a single ordinary path** on this CPU fixture. These paths have different function classes; no same-quality conclusion is possible.

**Files:** [full Japanese analysis](../../../../docs/phase2/MIRROR_DIRECTION_ATTENTION_HYPOTHESIS_2026-10-16.md), [benchmark](pilot.py), [4 mechanism tests](test_mechanism.py), [measured CSV](benchmarks.csv), [verification](VERIFICATION.json).

Reproduce in a new directory using PyTorch 2.10.0+cpu and Python 3.13:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m pytest -q test_mechanism.py
python pilot.py
```

**Scientific boundaries:** This is an unfused CPU forward diagnostic with random static view codes, not a trained Transformer, language model, complete FFN, full QKV projection, independent view quality comparison, KV-cache benchmark, or GPU test. Its main value is distinguishing computational feasibility from actual learning benefit.

Nearby prior studies include Axial Attention (https://arxiv.org/abs/1912.12180), Talking-Heads Attention (https://arxiv.org/abs/2003.02436), HotMoE (https://ojs.aaai.org/index.php/AAAI/article/view/40383), and the repository's MA-041. The new scientific question is whether a cheap Mirror-coordinate **m** adds marginal value beyond those native controls at matched quality, actual bytes, and active compute.
