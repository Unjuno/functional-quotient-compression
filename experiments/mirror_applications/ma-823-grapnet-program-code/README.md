# MA-823 — graph topology as program code

## H / T / D / C / U

**H:** Graph-program Views over shared node functions represent multiple topologies at lower bytes/active work than independent graph networks and improve on native sparse routing metadata.

**T:** Synthetic 8-node, 4-feature graph programs; eight directed topologies with 12 edges each; two message-passing layers; seed 82301; two programs separately scored; no training. Compared independent dense adjacency, native sparse edge indices, Mirror program code and one shared fixed topology.

**D: FAIL.** Dense adjacency with shared node functions uses 799 B and exactly reproduces outputs. Native sparse edge indices and Mirror program codes use 857 B each, with exact outputs and identical 12-edge execution. The fixed topology uses 727 B but has heldout nRMSE .785. Native sparse and Mirror payloads differ only by NPZ key name and hash, not structure or function.

**Evaluation boundary:** The two scored topology IDs and their complete edge lists are stored in inference payloads, so this does not establish topology generalization.

**C:** Ordinary graph routing metadata is the same program representation; for this small graph dense adjacency even serializes more compactly.

**U:** Learned graph programs, GrapNet, natural tasks, topology inference and fresh seeds. Fresh remained sealed.

## Evidence separation

- **Fact:** Payload bytes/hashes, output errors, edge counts, and replay are in `RESULTS_CORE.csv`.
- **Interpretation:** Topology codes are functional addresses but no advantage over native connectivity metadata is shown.
- **Hypothesis:** Larger sparse graphs may make edge lists cheaper than dense adjacency, but a Mirror-specific gain still requires a better code or execution path than native sparse routing.
