# MA-1171 — Mirror/Loop Sparse Autoencoder feature separation

Exploratory research branch. **This is an activation-disentanglement experiment, not an audio encoder.**

One 32-coefficient core generates 64 logical SAE detector vectors via 8 deterministic page views and learned 6D feature addresses. A three-pass sparse residual loop refines the detected features. The nonlinear Mirror decoder *of detector weights* is compared with an exactly budget-matched affine generated detector plus dense TopK SAE, dense iterative SAE, and rank8 learned-factor iterative SAE. The SAE *reconstruction dictionary* remains a fully learned explicit 64x32 object for all models, so no claim of compressing the full dictionary is allowed.

Synthetic 64-factor activations in 32 dimensions, 3 true active features per input. Independent and correlated-pair training worlds; held-out decorrelated feature mixtures assess out-of-distribution separation. True factors are **never supplied to training**. Evaluation aligns the learned dictionary to the planted factor dictionary using a one-to-one assignment and reports support F1, dictionary cosine and NMSE. The decoder correspondence is **not ground truth for real LLMs**.

See `PROTOCOL.json` for preregistered splits, train/dev/fresh discipline, controls, and strong success criterion. Freeze prior to train, preserve FAIL if Mirror does not beat identical-address affine and dense iterative controls. CPU only; no GPU or LLM claims.

References: Gregor & LeCun (ICML 2010); Gao et al (ICLR 2025); Karvonen et al (ICML 2025); Makelov et al (ICLR 2025).
