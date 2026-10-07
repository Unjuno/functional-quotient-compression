# MA-111 — semantic-role Mirror embedding

Status: SCREENING  
Branch: `research/ma-111-semantic-role-embedding-20261007`  
Base commit: `6e0ed10a094f9a082a144a2d8d86f508fd4cbad6`

## H — falsifiable hypothesis

One shared physical token embedding table and next-token readout, plus one small learned Givens coordinate per semantic role, can preserve role-conditioned next-token distributions on unseen lexical fillers with lower actual inference payload than private role maps. Roles whose transformations are not in that Givens family should need richer or private maps.

## Prior-art delta

PA13, *Attention as Binding: A Vector-Symbolic Perspective on Transformer Reasoning* (arXiv:2512.14709), interprets queries/keys as role cues and values as fillers and proposes explicit binding/unbinding heads and hyperdimensional memory. Zhang and McCoy (arXiv:2608.29034) fit Tensor Product Encoders to subject/verb/object representations, reporting R² above .60 on the sampled LLM representations and above .90 on embedding-model representations. These works establish role/filler structure and a TPR-style control; they do not test a compact per-role input-embedding view against tied embeddings and role-specific private maps.

Direct embedding controls also matter. *Weight Tying Biases Token Embeddings Towards the Output Space* (arXiv:2603.26663) reports tied input/output matrices can favor output-space geometry and discusses the capacity-versus-storage tradeoff. *Kronecker Embeddings* (arXiv:2605.29459) replaces input embeddings with fixed byte-position encodings plus a learned projection and reports a controlled three-seed nanoGPT result; its input-only replacement requires an untied output head. Aggregate Semantic Grouping (arXiv:2509.17737) composes static token representations from shared semantic concepts. These are adjacent embedding-compression methods, not role-conditioned Mirror views.

Cheapest controls are ordinary role-vector addition and FiLM, followed by per-role rank-2 input residual and per-role rank-2 output-head LoRA, and a per-role full linear output head (a direct role-conditioned unbinding/readout control). The rank-2 head update can represent a Givens-induced output change. A fixed sign-binding role code implements an explicit VSA-style control. Full per-role input maps are the private-function upper control.

## T — protocol

A controlled one-step next-symbol task uses 32 filler token IDs, four explicit roles (agent, patient, instrument, location), 16-dimensional shared input embeddings, and an 8-token continuation distribution from one shared readout. The shared embedding/readout are frozen after initialization and charged to every method. Development and evaluation split lexical fillers: train on IDs 0–23 and measure transfer on IDs 24–31 with fresh Gaussian input noise.

In the aligned family, each role teacher is the same shared function after a Givens rotation of the first embedding-coordinate pair. In the independent family, each role uses its own random orthogonal embedding transform. Distill temperature-2 teacher distributions for 128 examples/role and 300 updates/role. Development seeds are 11101/11102; learning rates .003/.01. Fresh seeds 11111–11113 remain sealed unless both development worlds pass the gate.

Controls: hard tying; additive role vectors; role FiLM; per-role rank-2 input LoRA; per-role rank-2 output-head LoRA; per-role full output heads; fixed VSA sign binding; learned full per-role input maps; and exact teachers as quality upper references. Final-role throughput is measured on a 2048-example batch after 5 warm-ups and 50 timed CPU forwards with one torch thread.

## Gates

Fresh aligned PASS requires each seed to meet all of: held-out teacher KL <=1e-4; top-1 agreement >=.99; ECE no more than .02 worse than the full-map control; total actual serialized inference payload <=.60x full per-role-map payload; and Mirror KL <= rank-2 output-LoRA KL + 1e-4. Report exact byte ratios and all other controls. A simpler control that matches quality at lower/equal bytes and compute blocks a Mirror-specific advantage.

If both development seeds miss either the absolute-KL/top-1 gate or the <=.60x full-map byte gate, fresh seeds remain sealed. This is a fixed-update representation screen, not a natural-language capacity or LM claim.
