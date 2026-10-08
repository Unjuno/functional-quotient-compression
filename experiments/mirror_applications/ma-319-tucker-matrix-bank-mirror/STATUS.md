# MA-319 status

- Status: **FAIL** under both frozen fresh gates.
- Dedicated branch: `research/ma-319-tucker-matrix-bank-mirror-20261008`
- Original protocol commit: `85246ca`; audit isolation amendment A1: `dbeec29`.
- Fresh seeds: 31911, 31912, 31913; Gutenberg audit SHA-256 `3f6bb9d6f78e0293b56acd4714dd68cb7d6d1d293402031ce9d5a216bcaf9d75`.
- Verification: 5 tests passed; 25 serialized packages reloaded; payload hashes exact and every NLL replay matched with zero difference.
- An earlier implementation bug evaluated the original Tiny Shakespeare audit split during development. Those logs are quarantined in `protocol_variants/contaminated_pre_amendment/` and excluded from all claims.
