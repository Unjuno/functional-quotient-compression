# MA-076 status

- Status: PROMISING
- Branch: `research/ma-076-one-block-many-layers-20261007`
- Base commit: `ccf4d5c4e83992d70ccdc5db6032e428f6532380`
- Last verified commit: pending
- Development complete: yes; seeds 76001–76002
- Fresh/audit opened: yes; seeds 76011–76013, after development gate
- Results committed: pending
- Verification committed: pending
- Registry row updated: pending

## H / T / D / C / U

- **H:** A compact per-depth Mirror coordinate can recover four aligned layer functions from one shared physical block, saving payload against independent layers and outperforming simpler shared controls at equal updates.
- **T:** Four 8×8 linear maps; tied, rank-1 static LoRA, per-depth generated gain, Mirror Givens conjugation, and untied controls. Two dev plus three fresh seeds on aligned and independent teachers; 600 Adam updates at LR 0.01; exact serialized inference payload, MAC proxy, wall time, and layer-output MSE recorded.
- **D:** PROMISING for the aligned mechanism screen: all three fresh worlds passed the 1.10x untied/shared quality gates and 0.80x untied-payload gate. Independent layer functions were not recovered. CPU runtime regressed.
- **C:** The aligned teacher was generated using the same Givens-conjugation family as the Mirror model; this favorable construction may explain the near-zero error. A static rank-1 LoRA control did much better than other shared controls on the aligned task, although at 17.1% larger payload than Mirror.
- **U:** Nonlinear Transformer blocks, natural-language NLL, near-convergence/capacity, optimized kernels, and generalization to held-out depth counts.

**FACT:** 3/3 fresh aligned quality/storage gates passed; 2/2 dev gates passed. Mirror payload 2,149B vs untied 2,729B (-21.3%); fresh independent MSE 0.684–0.762 vs untied near zero. Eager CPU training throughput was 0.235x untied on the aligned world. Tests 3/3 passed; 50 rows replayed with exact bytes and max MSE delta 4.68e-11.

**INTERPRETATION:** A small depth coordinate can convert one physical block into multiple useful logical maps when the maps follow that coordinate structure. This demonstrates a fixed-budget mechanism/storage frontier, not greater unconstrained capacity.

**HYPOTHESIS:** Depth views will lose their advantage as layer functions depart from the shared conjugation orbit; private residual parameters should recover quality at a measurable byte cost.

## Next action

Update registry/claim/status trackers, commit the result, and hand off to MA-079.

## Blockers

None. The experiment is limited to a synthetic CPU screen by design.

## Decisions / rulings

Protocol fixed before development evaluation. Fresh seeds were kept sealed until both development worlds met the quality and byte gates.
