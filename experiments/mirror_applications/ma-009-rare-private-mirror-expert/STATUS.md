# MA-009 status

- Status: FAIL under preregistered quality gate
- Branch: `research/ma-009-rare-private-mirror-expert-20261007`
- Base commit: `f45deaeccb` (full SHA in protocol)
- Development complete: yes; selected LR 0.01
- Fresh/audit opened: yes; worlds 90001–90003
- Results committed: yes (`e2232f0382eb66721e90d7b295fb0b2c33eb9731`)
- Verification committed: yes (`e2232f0382eb66721e90d7b295fb0b2c33eb9731`)
- Registry row updated: yes (tracker commit pending)

## H / T / D / C / U

- **H:** one private rare-role matrix plus common Mirror views recover aligned roles with fewer bytes; all-shared views fail at the outlier.
- **T:** synthetic hard-routed linear MoE, five controls, skewed role distribution, three fresh worlds, 1,200 updates.
- **D:** FAIL: storage gate passed (0.774x bytes), but full-MoE relative quality failed in two of three fresh aligned worlds; independent-all also required more capacity.
- **C:** rare-role relative errors are small in absolute terms; fixed-update and initialization variance may explain the threshold misses.
- **U:** longer updates, rare oversampling, nonlinear experts, and optimized kernels.

**Fact:** 30/30 rows replayed with exact bytes, four tests passed.
**Interpretation:** one private expert strongly repairs the rare role versus shared-only controls but is not reliably full-MoE-equivalent in this screen.
**Hypothesis:** a separate oversampling/compute-matched study could improve rare-role stability; fresh data must not be reused.
