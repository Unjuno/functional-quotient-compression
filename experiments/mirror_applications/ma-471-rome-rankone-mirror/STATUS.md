# MA-471 status

- Status: PROMISING, aligned synthetic rank-one storage screen
- Branch: `research/ma-471-rome-rankone-mirror-20261009`
- Base commit: `c935a903`
- Protocol freeze: `9dbb08b0`
- Development complete: yes
- Fresh/audit opened: yes, after protocol freeze
- Results committed: yes (pending report commit SHA)
- Verification committed: yes (pending report commit SHA)
- Registry row updated: yes (same report commit)

## Next action

Update the MA registry and claim ledger, validate integrity, and push the dedicated branch.

## Limitations

The preregistered private-residual control was not executed; results apply to aligned-only keys/deltas. This limitation is retained in U and no off-orbit claim is made.

## Decisions / rulings

- An initial fresh serializer retained oversized tensor backing storage through views; those rows were replaced before evaluation with cloned payload tensors, then all frozen fresh worlds/seeds were rerun. Only clone-backed rows are retained.
- The exact 2D key/value planes and angle coordinates are known by construction; no basis discovery is claimed.
- Apply latency is a small CPU microbenchmark and does not establish Transformer serving throughput.
- Actual serialized payloads include shared planes, angles/coefficients, scales, metadata, and the standalone ROME factors.
