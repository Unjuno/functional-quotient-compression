# MA-997 — Structured sparse equivariant support × Mirror code

Status: **FAIL for Mirror-specific value in the scoped synthetic mechanism screen**. Official PA300/MACE adaptation remains **NOT ESTABLISHED**.
Branch: `research/ma-997-sparse-support-mirror-20261113`

## H / T / D / C / U

**H — Hypothesis:** a shared six-coordinate update support with per-material Mirror coefficients recovers multiple conservative material residuals with at least 25% fewer actual bytes than independent sparse adaptation and beats the matched native shared-support control.

**T — Test:** Reused the MA-996 CPU radial energy/force generator only. Development seed 99701 selected six common coefficient indices from 12 synthetic reference materials ×20 configurations. Frozen support was [0, 4, 8, 12, 16, 20]. Fresh worlds 99711/99713/99717 each generated four aligned target residual potentials, with eight calibration and 24 held-out configurations per target. Compared shared-only, Mirror shared support, matched native shared-mask coefficients, per-target independent sparse masks selected from calibration, and independent full coefficients. Zero optimizer updates. All complete deployment payloads, including shared base/basis, mask metadata, and coefficients, were serialized to NPZ. Batch CPU wall times were 0.511s, 0.150s, 0.447s; they are aggregate run times and no per-method runtime conclusion is drawn.

**D — Decision:** **FAIL**. Mirror and native shared-support control were numerically identical in all worlds, with identical 1,732 B serialized payloads. The payload was only 36 B (2.0%) below independent sparse masks (1,768 B) and 42 B (2.4%) below independent full coefficients (1,774 B), below the preregistered 25% gate. The six-value material decode adds six MACs per material; per-configuration radial inference proxy is 100 MACs for all methods.

**C — Strongest counter-hypothesis:** the benefit is ordinary support sharing plus sparse coefficients, fully represented by the native shared-mask table. The teacher itself had perfectly aligned sparse support, which is a favorable synthetic condition.

**U — Unconfirmed:** no official MACE-MP/MPTrj, learned sparsity-promoting adaptation, natural materials, long MD, or unseen support shifts were tested. The independent sparse-mask selector from only eight calibration examples missed the known support and had mean combined nRMSE 0.155; this does not imply that a trained PA300 selector would fail.

## Facts

- Mirror/native shared-support combined nRMSE averaged 3.97e-16 across fresh worlds; held-out energy and force errors were at floating-point precision.
- Independent sparse mean combined nRMSE was 0.155; independent full mean was 0.038 (one world had .115 due sparse calibration/conditioning).
- Maximum finite-difference conservative-force residual: 1.11e-10; maximum rotation-equivariance residual: 2.01e-16. All 15 payloads round-tripped and the metrics replayed exactly from serialized states (max difference 0).
- Across two consecutive atomistic material-code screens, Mirror coordinates exactly matched the native low-rank or native shared-mask control. Pause MA-998's direct species/environment coefficient-factor lane pending a structurally distinct insertion; continue to MA-999, a different registered family.

## Interpretation

A shared sparse coordinate can reduce mask metadata slightly when support is perfectly aligned, but the measured saving is too small to pass the frozen gate, and the same representation is ordinary shared-support sparse adaptation. This screen supports no Mirror-specific claim.

## Hypothesis for follow-up

MA-999 should test temperature × pressure response as an independent condition-factor mechanism with native MatterSim conditioning and scalar/FiLM controls. It must preserve energy-gradient force consistency and count all condition codes and shared states.
