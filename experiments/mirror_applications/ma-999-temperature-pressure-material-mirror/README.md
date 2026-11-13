# MA-999 — Temperature × pressure material Mirror factor

Status: SCREENING. Official MatterSim weights and reference ensembles are unavailable. This screen tests a narrower physical boundary: at fixed atomic coordinates, a potential energy/force law does not become a different function merely because its thermodynamic sampling condition changed.

## H / T / D / C / U

- **H:** A compact T×P code improves heldout force/energy performance beyond a shared condition-invariant potential and native direct condition scalars.
- **T:** Synthetic Lennard-Jones pair energy/forces sampled at T/P-dependent box scales; three locked fresh seeds; shared-only, separable Mirror factors, and native direct T/P scalar table.
- **D:** Pending.
- **C:** The shared physical potential already encodes the force law; T/P only changes the distribution of sampled geometries. Native condition controls also represent any variation exactly.
- **U:** MatterSim T/P generalization, real NPT ensemble fidelity, MD stability, and material transfer remain untested.

PA299 establishes MatterSim as a broad atomistic foundation model across materials and thermodynamic conditions. This screen is synthetic and does not reproduce PA299.
