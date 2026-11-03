# Amendment 4 — measure runtime instead of reporting a zero placeholder

The first two valid development runs reported per-method `encode_seconds` as a constant zero, despite the frozen protocol requiring payload-encoding time. Their quality and serialized-byte metrics remain diagnostic and are not used as final registered results. This amendment adds measured model-load, candidate error-table, fit, per-method quantize/serialize, and total wall times. It changes no candidate selection, reconstruction, payload or gate logic. Both registered development seeds are rerun from this source before the verdict; previous outputs are retained externally but excluded.

Source SHA-256 before: `c0c39bd048115ac334b9c3fc2c5f106489c9a906435d8edb1641c1a61151d88e`
Instrumented source SHA-256: `0cc73df55feefa0ffb625a2e41ea20a07ed1da8d309e1f401840a538bf5a2bec`
Protocol SHA-256 unchanged: `0754daffab4540278d8e96afb4d9d774c4584115a46de94024db0e80cceb5c94`.
