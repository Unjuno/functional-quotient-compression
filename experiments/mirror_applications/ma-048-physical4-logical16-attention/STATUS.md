# MA-048 status

- Status: FAIL under preregistered aggregate output-quality gate
- Branch: `research/ma-048-physical4-logical16-attention-20261007`
- Development complete: yes; LR 0.003
- Fresh/audit opened: yes; worlds 48001–48003
- Results committed: yes (`bdd1dd3002b493cf1355dcc297e9de23b5e0b532`)
- Verification committed: yes (`bdd1dd3002b493cf1355dcc297e9de23b5e0b532`)
- Registry row updated: yes (tracker commit pending)

## H / T / D / C / U

- **H:** four physical QKV groups plus sixteen head views recover an aligned teacher with lower payload.
- **T:** synthetic 32D attention, four methods, aligned and independent teachers, three fresh worlds.
- **D:** FAIL: storage passed (0.565x bytes), output-MSE gate failed 3/3; head-local contribution audit was better but aggregate output and runtime were worse.
- **C:** fixed-update optimization or output-head compensation may account for the gap.
- **U:** longer schedules, fused kernels, causal attention, NLL and capacity.

**Fact:** 24/24 rows replayed with exact bytes; tests 3 passed.
**Interpretation:** per-head structure alone did not recover combined attention behavior.
**Hypothesis:** optimized view execution might improve runtime and quality; untested.
