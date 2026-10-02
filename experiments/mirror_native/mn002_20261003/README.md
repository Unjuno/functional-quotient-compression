# MN002: learned shared-state causal Transformer

**Status: work in progress, CPU research prototype.** Phase II of Unjuno's Vector Mirror / FQC project. This is native conditional computation, not a new quality-preserving 64x compression result. No pretrained weights, remote compute, paid service, or distillation are required.

## What was tested

Two causal Transformer layers, model width 32, four attention heads, 32-wide shared FFN, and 1/4/16 soft state vectors per layer. The state bank, shared matrices and router are trained jointly. Attention is ordinary causal attention. The FFN computes a weighted state vector and modulates shared hidden features; it never constructs a token-specific expert matrix. **This is soft feature modulation, not sparse expert dispatch or a literal geometric reflection/Fourier transform.** Ordinary MoE routers and experts are also trained jointly; joint training alone is not our novelty claim.

The task has 16 random permutations of 32 symbols (512 rule-symbol mappings). Each sequence contains four `[rule, symbol, answer]` records. Only the next-token answer slots are supervised. All rule-symbol pairs occur in training. Audit sequences are disjoint, but their mappings are familiar: **this is finite conditional-function fitting, not unseen-rule generalization or natural-language modeling.**

MN002: fixed table seed 271828; eight variants x seeds 201/202/203 = 24 runs. MN002R: a follow-up designed after MN002, with a fresh table 161803 and seeds 211/212/213; three variants = 9 runs. Do not treat 33 runs as 33 independent task distributions.

Training is fixed at 600 updates, batch 64, AdamW learning rate 0.003, betas (0.9,0.95), weight decay 0.01, gradient clip 1.0. All 24/9 final checkpoint hashes are locked before the corresponding audit is generated. Each audit contains 2,048 sequences / 8,192 answer positions. No best-audit checkpoint selection is used.

## Main results: all three seeds

Median answer-token NLL [full observed range], lower is better. Accuracy is not the capacity of the model.

| Variant | Parameters | Complete FP32 artifact bytes | NLL median [range] |
|---|---:|---:|---:|
| Mirror 1 state | 17,331 | 69,420 | 0.050240 [0.047465, 0.056821] |
| Matched Dense 33 | 17,331 | 69,420 | 0.064346 [0.037656, 0.070810] |
| Mirror 4 states | 17,721 | 70,980 | 0.036891 [0.021145, 0.044999] |
| Matched Dense 36 | 17,721 | 70,980 | 0.050127 [0.040854, 0.060516] |
| Mirror 16 states | 19,281 | 77,220 | 0.038157 [0.034471, 0.047844] |
| Matched Dense 48 | 19,281 | 77,220 | 0.030322 [0.025510, 0.030728] |
| Direct feature gate 32 | 19,313 | 77,348 | 0.027460 [0.017112, 0.037947] |
| Full soft-MoE, 4 independent FFNs | 30,137 | 120,644 | 0.015709 [0.008994, 0.018444] |

Mirror4 beats its exact-parameter Dense control in 3/3 seeds (paired relative NLL reduction 25.64% to 48.24%, median 26.40%). **Mirror16 loses to its exact-parameter Dense control in 3/3.** More states are not a monotonic performance guarantee. The full soft-MoE reference computes all four experts; it is larger and has more active linear computation, not a matched sparse-MoE baseline.

## Fresh-table follow-up: do not hide the weaker result

| Variant | Parameters | Bytes | NLL median [range] |
|---|---:|---:|---:|
| Mirror 4 states | 17,721 | 70,980 | 0.040750 [0.029401, 0.047707] |
| Matched Dense 36 | 17,721 | 70,980 | 0.035518 [0.033154, 0.081282] |
| Direct feature gate 24 | 17,745 | 71,076 | 0.036929 [0.030780, 0.043790] |

Mirror4 wins against Dense in 2/3 paired training seeds, but its marginal median NLL is worse. It wins against the simple direct gate in only 1/3. The gate uses 24 extra parameters (+0.136%) and 96 extra bytes. **Unique or general parameter-efficiency superiority is NOT demonstrated.** MN002R is a follow-up of a discovery-selected S4 configuration, not a comprehensive independent architecture search.

## Routing and cache evidence

Replacing the learned Mirror4 router with the input-independent all-position development-mean distribution lowers main-task accuracy from 99.12-99.78% to 54.27-71.72%. On the second table it drops from 99.33-99.60% to 58.58-71.72%. This shows that these trained models use their routing; it does not show routing is necessary for this task or that states are independent experts. A trained one-state model is an input-independent negative control, and freezing its route changes nothing.

Full forward and cached chunks of 1/3/5 tokens were compared on all 33 models: maximum logit error 1.62125e-5, all argmax values identical, below the frozen 5e-5 tolerance. FP32 cache cost is 512 bytes per cached token per sequence for every variant. These statements assume deterministic causal token-local routing, fixed learned weights, and no retroactive change to past token states. No cross-model cache reuse is claimed.

## Exact storage and compute accounting

`weights.py` implements MNW v1: a fixed 64-byte little-endian header, all FP32 inference parameters in architecture-defined order, and a 32-byte SHA-256 trailer. It includes token/position embeddings, norms, attention, FFNs, routers/state banks and the LM head. No hidden model-specific numeric parameters. Decoder/source and PyTorch runtime are common and outside the weight-file accounting. Activations, optimizer memory and KV caches are separate. This is FP32 native storage, not quantization.

All 33 `.mnw` files were loaded and their complete audit was replayed; every score and per-sequence NLL matched the original checkpoint audit exactly. The `.pt` training checkpoints have variable framework overhead and are not used for matched-file-byte claims.

At model/FF width32, Mirror S and Dense width(32+S) have exactly equal learned-scalar counts AND FFN linear MACs. Across both FFNs these are 4,224 / 4,608 / 6,144 MACs per token at S=1/4/16. Attention/head costs are common. This omits nonlinearity, router softmax, elementwise work, memory traffic and dispatch, so **total runtime is not matched or benchmarked**. Both direct-gate controls match the respective linear MACs but differ slightly in biases/parameter count.

## Reproduce locally

From this directory, using Python 3.13.5 and the versions in `requirements.txt`:

```bash
python -m pytest -q
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python reproduce.py --output /tmp/mn002-new-run
```

The output path must not already exist. No network connection, data download or API key is required once dependencies are installed. Individual stages are also available:

```bash
python train.py --name mirror_s4 --seed 201 --output /tmp/mn002-single
```

For full-suite freezing/audit and artifacts see `reproduce.py`. Small published summaries are in `results/`; raw checkpoints and per-sequence data are regenerable. The conversation evidence ZIP retains all 33 training checkpoints, all 33 complete `.mnw` models, full results, and original MN001 archive. Large generated outputs are not committed to normal Git history.

Tests: 41 passed. One primary run was retrained in a separate process and all numerical results (excluding runtime/environment measurements) and checkpoint bytes matched. This is same-environment reproducibility, not a cross-version/platform guarantee. Environment: Intel Xeon Platinum 8573C, observed 2.300 GHz (not locked), two numerical threads, PyTorch2.10.0+cpu; FP32 training/forward, FP64 reported cross-entropy. No speed benchmark.

## Limits and next useful tests

The two finite tasks still approach accuracy saturation. A fixed 600-update loss comparison mixes optimization and approximation effects; it is not asymptotic capacity. State-count comparisons also change initialization statistics. Three training seeds per condition are not a population theorem. Router entropy, state count and paths do not count independent functions. No natural-language performance, meaningful LLM scale, optimal sparse routing, Fourier-specific benefit, general novelty or unrestricted effective-capacity increase has been shown.

Next: non-saturating sequence tasks, additional independently chosen task tables, equal-training-compute frontiers, stronger direct-gate/FiLM/shared-expert controls, retrained static-routing controls and carefully measured runtime/memory. Publish null and negative results too. Do not tune on either final audit.

## Related work and reuse

- Shazeer et al., sparsely-gated MoE (2017): https://arxiv.org/abs/1701.06538
- Perez et al., FiLM (2017): https://arxiv.org/abs/1709.07871
- Wen, Tran and Ba, BatchEnsemble (2020): https://arxiv.org/abs/2002.06715
- Gao et al., Parameter-Efficient MoE (2022): https://arxiv.org/abs/2203.01104

This is a scoped experimental prototype, not a claim to have invented parameter-sharing MoE or feature modulation. Reuse and evaluation are welcome under the repository's existing **Apache-2.0** license. Please cite Unjuno's repository/experiment when building on it; `CITATION.cff` is supplied. Citation is a request, not an additional license restriction.
