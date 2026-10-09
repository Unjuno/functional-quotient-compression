# MA-332 A0 implementation defect

The initial A0 run was quarantined before deciding the candidate. `arrays()` promoted W1/W2 to float64 during division, while `payload()` declared float32. Reloading those payloads as float32 shifted code offsets and produced invalid coordinate values. The A0 fresh rows are retained for provenance but are not evidence. Amendment A1 fixes the dtype explicitly, verifies serialized payload decoding and replay, and uses unseen A1 development/audit worlds.
