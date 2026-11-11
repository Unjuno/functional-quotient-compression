# MA-876 — shared basis for cross-model KV translators

This PA236 screen tests whether per-head affine KV translators for multiple ordered source-target model pairs can share a low-rank translator basis and use small pair/layer/head codes, while preserving target next-token quality and reducing actual mapper bytes.

**Mirror insertion:** this experiment adds a Mirror coordinate at each affine per-head KV map interface so that multiple source-to-target cache translators can be reconstructed from one shared basis rather than storing each full map.

The models are local cached pretrained Pythia-70M, GPT-2, and SmolLM2-135M. The heldout ordered pair is Pythia→SmolLM2; the basis is fit using GPT-2→Pythia and GPT-2→SmolLM2. Native full ridge maps and native PCA map coding are required controls. This screen runs on CPU; it makes no GPU latency claim.
