# MA-999 — Temperature × pressure MatterSim Mirror factor

Status: **FAIL for Mirror-specific value in the scoped synthetic condition-invariance screen**. MatterSim/PA299 remains **NOT ESTABLISHED**.
Branch: `research/ma-999-temperature-pressure-material-mirror-20261113`

## H / T / D / C / U

**H — Hypothesis:** a compact T×P Mirror coordinate improves heldout condition-specific energy/force quality beyond a shared potential and ordinary direct condition table, while reducing actual bytes.

**T — Test:** Analytic five-particle Lennard-Jones energy and its exact negative-gradient forces. T/P grid: 300/600/900 K × 1/2/4 pressure units. Conditions changed only the sampled box scale using an ideal-gas-inspired volume ratio; this was not an NPT sampler. Development seed 99901 established that the target 3×3 scalar table is rank one; rank-1 T/P code was fixed in the committed runner before fresh evaluation. Fresh seeds 99911/99913/99917 used eight calibration and 24 heldout configurations per each of nine condition cells. Compared shared-only, rank-1 Mirror factors, and an ordinary direct T/P scalar table. 72 calibration and 216 heldout examples per world; zero optimizer updates. Serialized NPZ payloads charge all parameters, factors/table, condition values and metadata. Fresh batch CPU wall times were 0.165s, 0.171s, 0.344s; these are joint batch times and do not establish per-method latency.

**D — Decision:** **FAIL**. Shared-only has exactly zero heldout error in all worlds because the physical law is condition-invariant. Mirror's residual combined nRMSE is at floating-point precision, but its deployment payload is 1,618 B versus 526 B shared-only (+208%) and 1,396 B for the direct condition table. Its active proxy is 101 versus 100 MAC/config. No quality or storage advantage.

**C — Strongest counter-hypothesis:** the pressure and temperature labels select different sample distributions, not different energy functions at the same coordinates. The shared energy law already handles all conditions; the code is extra state with no useful function to represent in this test.

**U — Unconfirmed:** MatterSim inference across true thermodynamic ensembles, material phase fidelity, real NPT trajectories, OOD T/P generalization, and MD stability were not tested. The synthetic geometry sampling is only a constraint check.

## Facts

- Shared-only and direct native table errors were zero. Mirror combined error averaged 4.81e-16; its factor matrix remained rank one.
- Maximum same-coordinate force spread across condition labels: 2.78e-17; finite-difference force-gradient residual 8.91e-12; rotation-equivariance residual 4.16e-16.
- Actual deployment bytes: shared-only 526 B; native condition table 1,396 B; Mirror factors 1,618 B. Mirror factorization did not save actual NPZ bytes because all factors, grid values and metadata were charged.
- Serialized payload metric replay was exact in all three fresh worlds.

## Interpretation

This experiment rejects the claim that T/P labels alone create a need for separate force functions in the tested physical setting. It does not reject condition-specific sampling, phase prediction, or MatterSim's learned generalization across thermodynamic states.

## Hypothesis for follow-up

MA-1000's independent energy-gradient audit is the next P0 candidate. It should use a genuinely condition-dependent Mirror perturbation and compare force-from-energy against a separately parameterized force head before any MD stability claim.
