# MA-418 — compositional object/style function codes

## H — falsifiable hypothesis

Object and style Mirror factors learned from observed pairs will compose to reconstruct held-out pair functions, approaching a factorized ordinary latent decoder while using fewer bytes than pair-specific codes.

## T — protocol

Four object identities × four style identities define continuous functions. Train on 12 pairs and hold out the diagonal four pairs. The teacher composes object-specific and style-specific Givens views on a shared decoder. Compare factorized Mirror, factorized latent concatenation, factorized FiLM, direct pair-specific Mirror codes with untrained held-out entries, shared-only, and independent per-pair oracle decoders. Evaluate seen and held-out pair query reconstruction separately.

This is aligned compositional feasibility, not natural object/style transfer.

