# MA-199 status

- Status: **FAIL at development**; no fresh worlds opened
- Branch: `research/ma-199-gradient-coordinate-20261007`
- Protocol freeze commit: `6a2e22c08c3e766359c986e4b7202b4651feee39`
- Result commit: `577ceb9f5d8d3d55f8350808e03bb5ee3839e090`
- Development complete: yes
- Fresh/audit opened: no
- Results committed: yes
- Verification committed: yes (in tracker commit)
- Registry row updated: yes

## H — hypothesis

Task-specific Givens gradient coordinates on a paid shared plane will encode aligned updates with lower total bytes and optimizer resume state than per-task rank-1 LoRA while retaining old skills.

## T — execution

16x8 frozen base, shared 16x2 plane, four sequential tasks, 64 examples and 300 updates per task. LR .003 selected by registered dev metric over .01 on seeds 19901/19902. Controls: hard tie, generic `P @ C_t` shared plane, rank-1/2 LoRA, shared rank-1 hypernetwork, independent full.

## D — decision

**FAIL at development.** Mirror total inference payload was 1,103B versus 1,196B rank-1 LoRA (0.922x), missing <=0.90x on both development worlds. Fresh seeds 19911–19913 stayed sealed. The generic shared-plane control had nearly identical mean aligned MSE and smaller resume state (1,267B/skill vs 2,119B/skill). Mirror/LoRA rank-1 quality ratios were 18.10 and 0.656, unstable across seeds.

## C — strongest counter-hypothesis

The optimizer checkpoint stores angle and vector as separate parameters, adding metadata. A packed coordinate parameter could shrink real resume bytes. This does not change the failed total-inference-byte gate or the generic-control result.

## U — unresolved

Fresh quality and retention were not established. Basis discovery cost from real gradients and language-model performance remain untested.
