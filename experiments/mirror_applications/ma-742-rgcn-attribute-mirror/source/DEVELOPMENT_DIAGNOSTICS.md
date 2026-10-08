# Development run provenance

`development_150_updates.json` retains the initial 150-update screen made with generic `torch.save` serialization. `development_600_torchsave_diagnostic.json` retains the same 600-update model runs with generic pickle payload sizes. These files document the optimization and serialization checks; their payload byte columns are not authoritative.

`development_600_updates.json` is the frozen development comparison using the versioned compact inference payload described in `PAYLOAD_FORMAT.md`. It is the file hashed in `frozen_config.json`. No fresh data was opened when selecting rank 2 and 600 updates.
