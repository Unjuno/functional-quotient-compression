# A8b — post-hoc optimization diagnostic (NEW worlds)

Written **after seeing A8's audit**, before training any A8b models. This is exploratory hypothesis repair, not confirmatory preregistration.

Question: can longer gradient training (2,400 vs A8's 600 updates) remove the 512-byte ramp-family gap for Mirror? The A8 exact construction proves the family is representable at only 244 bytes; that proof does not establish training will discover the solution.

Fixed A8b scope: new worlds 816,817,818,819; only `ramp` family; 2 seeds each (0,1); single cap 512 B; variants `mirror`, `pre`, `film`, `shift`; 2400 AdamW updates, lr=.012, batch256, gradient clipping5, no early stopping; 4 workers CPU FP32; 4×2×4=32 models. No tuning, no teacher parameters exposed to training; lock checkpoint hashes before any val/IID/OOD evaluation; audit on the frozen models. Same A8 learner, teacher and serializer. Compare world-level IID/OOD MSE ratios and count worlds meeting IID/OOD NRMSE<=.08. Compare these 4 new worlds against A8 descriptively only; they are not paired training continuations from the A8 checkpoints, so no causal estimate isolating step-count from world variation.

H: mirror at 2400 updates reaches joint NRMSE<=.08 in >=3/4 fresh worlds and beats validation-selected `pre`/`film`/`shift` in >=3/4 worlds. D: if 0/4 or below comparator, FAIL on tested optimization protocol; if positive, exploratory only, require same-world fixed-step replication. C: higher width/activation patterns and optimizer geometry may block Mirror even if representable. U: four worlds and 2 seeds, finite time, no equal FLOPs, choice of A8b family came after A8; any t-interval is descriptive.
