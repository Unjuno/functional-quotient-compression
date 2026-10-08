# MA-314 pre-fresh amendments

Both changes were recorded before opening any fresh seed.

1. “Selects each aligned group’s true dimension” is evaluated by group median because one low-amplitude 4D task fell below validation threshold and selected 2D at the strictest candidate. Per-task selections remain in `ALLOCATION_EVENTS.csv`. The selected threshold is 1e-4; both dev seeds have aligned group medians 2/4/8/16 and mean normalized test MSE <=1e-4.
2. The active-inference operation proxy now uses actual selected dimensions rather than treating every adaptive task as max-dimension. The earlier dev summaries are retained at `protocol_variants/pre_active_ops_correction/`; this correction changes neither fit nor quality nor serialized bytes.

The matched adaptive-direct payload was smaller than Mirror in both development seeds. Locked fresh seeds are nevertheless run under the frozen protocol to verify the result.
