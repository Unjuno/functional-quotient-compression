# MA-487 status

- Status: SCREENING
- Branch: `research/ma-487-lista-sparse-function-inference-20261008`
- Protocol frozen: no
- Development complete: no
- Fresh/audit opened: no

## Next action

Freeze LISTA architecture and training conditions, then run development seeds 48701 and 48702 only.

## Decisions / rulings

This candidate measures learned sparse-code inference against direct projection and OMP. The presented function matrix is an encoder input, not part of serialized predictor state; do not claim that predictor bytes compress an unseen function bank without charging its inputs or stored codes.
