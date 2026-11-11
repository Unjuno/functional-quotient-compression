# MA-823 status

- Status: FAIL for Mirror-specific attribution and storage frontier.
- Branch: `research/ma-823-grapnet-program-code-20261111`
- Base: `99527143`
- Development: synthetic eight graph programs; 8 nodes; 12 edges per program; two message-passing steps.
- Fresh/audit: sealed after exact native sparse-index alias.
- Verification: focused test passed; payload replay exact.

## H / T / D / C / U

- **H:** Graph-program code reduces architecture storage/compute and generalizes to heldout topologies.
- **T:** Synthetic graph substrate; dense adjacency, native sparse edge indices, Mirror code, fixed topology.
- **D:** **FAIL.** Mirror/native sparse both 857 B and exact; shared node functions plus dense adjacency are 799 B. Fixed topology misses heldout quality (.785 nRMSE). The scored topology lists are in payload, so OOD topology was not tested.
- **C:** Native connectivity metadata fully explains the program coordinate; dense adjacency is smaller at this graph scale.
- **U:** Learned graph architectures, larger graphs, natural tasks and fresh seeds.
