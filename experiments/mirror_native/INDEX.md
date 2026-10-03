# Mirror-native experiment index

This directory indexes the public experiment lane. Large local evidence bundles are not committed to normal Git history.

| Experiment | Main question | Canonical status |
|---|---|---|
| MN001 | Can a tiny causal Transformer learn input-dependent shared states? | Feasibility only; Dense also solves the toy. |
| [MN002](mn002_20261003/README.md) | Do a few learned shared states help at matched resources? | Mixed result; small positive case, no general advantage. |
| MN003 | Can independent block routers be packed and executed in parallel? | Numerical equivalence and CPU loop-overhead reduction demonstrated; capacity advantage unresolved. |
| MN004 | Does overlapping sparse support plus averaging help? | Learned supports work, but averaging is not consistently better; fixed-support sum/mean are reparameterization-equivalent under stated assumptions. |
| MN005 | Can where-routing be restricted to a candidate bank? | Lower parameter/file cost; runtime benefit depends on shape; quality is condition-dependent. |
| MN006 | Is router early lock-in the main failure mode? | Strong lock-in explanation not supported; router-only tuning helps, joint tuning helps more. |
| MN007 | Can semantic concepts share a parameter identity and differ by a small Mirror state? | Compact representation demonstrated; strong semantic self-organization and Mirror-specific advantage not demonstrated. |

See [Phase II research state](../../docs/phase2/RESEARCH_STATE_THROUGH_MN007.md) for quantitative findings and evidence boundaries.

## Local evidence hashes

| Experiment | ZIP SHA-256 |
|---|---|
| MN003 | `a804ebd3b2c2aef37b864686711a54024105d1149d4eaebb816d9136d4fc2fc4` |
| MN004 | `270fe2f34c211ad51809a8761a6a4432c0a6b8a91a5725ab95eb18546f5446ea` |
| MN005 | `23c0273a849ec5037f110688c29882ce0e99e9965e2d43a113446576c42488c7` |
| MN006 | `a7c6bd978ad48b2b41624653b46d0fc39d921060542dfa191b91b43ae2144846` |
| MN007 | `41e6359eeff530b9f57e56b3ed91b887c2cb6dfcb4e3e2c5c0a3f97076d194df` |

These hashes are provenance identifiers only. The binary packages are intentionally not stored in ordinary Git history.
