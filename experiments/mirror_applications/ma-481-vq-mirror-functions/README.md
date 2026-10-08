# MA-481 — VQ Mirror address codebook for logical functions

Status: SCREENING  
Branch: `research/ma-481-vq-mirror-logical-functions-20261008`  
Base commit: `58ed65e`  
Prior art: PA91 (VQ-VAE)

## H — Hypothesis

A discrete address can select one latent coefficient vector and shared function decoder to produce many logical functions from a small physical matrix-atom bank. Compare query behavior and actual complete bytes with independent functions, continuous coefficients, native VQ and int8.

## T — Frozen protocol

The bank has 192 linear functions over 16 dimensional inputs/outputs. Fit four shared matrix atoms and codebooks on 128 function matrices; evaluate the 64 heldout functions on common query inputs. VQ sizes are 16/64/128. Fresh seeds 48111–48113 remain sealed unless all gates pass. See `PROTOCOL.json` for codebook fit, payload and operation definitions.

PA91 uses discrete latents to select shared decoder outputs. MA-481 measures function-level heldout behavior and address collisions rather than treating the number of possible code sequences as independent capacity.

## Results

Pending frozen development runs.
