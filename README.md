# Vector Mirror Research — FQC and Mirror-native models

**Research in progress. No quality-preserving 64x Transformer result or general capacity advantage is claimed.**

## Current organization (2026-10-03)

| Phase | Scope | Status |
|---|---|---|
| [Phase I: Functional Quotient Compression](docs/phase1/STATUS.md) | Post-training compression, task-sensitive shared/private codecs, exact byte accounting | **Paused; results, code and history preserved** |
| [Phase II: Mirror-native conditional models](docs/phase2/README.md) | Jointly trained shared FFNs, small state vectors and input-dependent routers | **Active; minimal CPU experiments** |

The previous README is preserved verbatim in [the Phase I snapshot](docs/phase1/README_before_phase2_20261003.md). Its main-branch baseline was `41e084440e5c8525c1e1adeefd172be7f05acb10`. No historical source, tests, claims or experiment directory was deleted by this transition.

## Start here

[**MN002: runnable code, fixed protocols, positive and negative results**](experiments/mirror_native/mn002_20261003/README.md)

The small causal Transformer learns input-dependent shared states without expanding a full expert weight matrix per token. MN002 compares 1/4/16 states against exactly parameter-, file-byte- and FFN-linear-MAC-matched Dense controls. It includes 24 main training runs and a 9-run fresh-table follow-up, plus cache and serialized-model tests.

**Result boundary:** Mirror4 beats matched Dense on the first finite task in 3/3 seeds, but only 2/3 on the second task and only 1/3 against a near-matched simple gate. Mirror16 loses to its Dense control in 3/3 primary seeds. This supports feasibility and motivates further testing; it does not demonstrate universal efficiency, exponential capacity, natural-language quality or a new invention of parameter-sharing MoE.

## Reproduce

```bash
cd experiments/mirror_native/mn002_20261003
python -m pip install -r requirements.txt
python -m pytest -q
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python reproduce.py --output /tmp/mn002-reproduction
```

After dependencies are installed, the experiment requires no network, model download, API key or paid service. Select a new output directory. Full protocols and limitations are in the experiment README.

## Reuse and evaluation

Code and extensions are welcome under the existing **Apache License 2.0**; see [LICENSE](LICENSE). Please cite Unjuno's repository/experiment when using it or reporting evaluations. A [citation file](experiments/mirror_native/mn002_20261003/CITATION.cff) is provided. Citation is a request, not an additional license restriction.

Independent evaluations, including null/negative results, are welcome. Useful targets are non-saturating tasks, stronger matched-resource gate/shared-expert baselines, cache correctness and real throughput/memory. The project is unfinished, and the public prototype is not a production model.
