# MA-912 StableVC source and asset review

- PA258 is StableVC, arXiv:2412.04724, accepted by AAAI 2025. It decomposes content, timbre, and style/prosody and uses conditional flow matching with gated dual attention.
- The arXiv HTML links to `https://yaoxunji.github.io/stablevc/`; that page currently serves unrelated stale NeuralDPS HTML. GitHub repo search did not surface an implementation. A lowercase/uppercase direct repo check found `yaoxunji/stablevc` commit `6c1c162840958497633991a2e6b2fcc0467bea5f`, but it is a 59 MB project page with README `# stablevc`, HTML and WAV/mp4 examples only; it contains no model source, training/evaluation scripts or checkpoints.
- Example audios do not provide a runnable codec/VC model or complete heldout evaluation data. No StableVC code/checkpoint is available in the workspace.
- Current runtime is CPU-only PyTorch 2.14.1 and no general VC pretrained source/checkpoint/assets. No local licensed/consented speech corpus is present.

No model, feature extraction or Mirror/StableVC comparison can be run faithfully. A different voice conversion model would change the registered task.
