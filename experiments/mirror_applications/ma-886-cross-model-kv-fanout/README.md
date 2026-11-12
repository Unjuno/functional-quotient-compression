# MA-886 — one source KV prefill to multiple targets

This experiment tests whether one Pythia-70M producer cache can feed both Pythia-160M and SmolLM2 targets with one shared affine translator plus compact target-role views. It counts producer prefill once, measures both target qualities and charges the complete serialized mapper state.

Prior art PA236/PA239 already covers cross-model KV transfer and prefill reuse. This experiment does not claim to reproduce those systems. See [PROTOCOL.json](PROTOCOL.json) for the frozen data split, controls, quality/latency/byte gates, and limits.
