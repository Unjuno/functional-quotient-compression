# MA-649 — Meta-SGD Mirror update vector

Status: SCREENING. Test whether family-specific Meta-SGD update vectors share a low-description basis when task families have related diagonal curvature.

The synthetic task uses a 32D shared initialization and eight quadratic task families with different curvature patterns. Fit global Meta-SGD, independent family vectors, rank-4 Mirror/PCA codes, and a random-subspace control using support gradients; evaluate one-step query loss and actual serialized inference bytes across development task worlds. Fresh seeds 64911–64913 stay sealed unless the preregistered gate passes.

See `PROTOCOL.json`.
