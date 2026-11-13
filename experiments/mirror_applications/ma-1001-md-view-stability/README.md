# MA-1001 — Mirror force-code rollout stability

Status: SCREENING. PA297/PA299 motivate long-horizon stability checks because one-step accuracy does not guarantee trajectory quality. No official model or material data is available; this is a synthetic NVE integrator screen only.

## H / T / D / C / U

- **H:** A fixed energy-layer Mirror code remains stable over long NVE rollouts when forces are derived from energy; native parameter alias should match exactly.
- **T:** CPU Lennard-Jones velocity Verlet; 3 fresh worlds, 2 task codes, 3000 steps, native per-task coefficient alias and negative control.
- **D:** Pending.
- **C:** Stability follows from integrating the same physical potential with the same fixed timestep; it is not uniquely a Mirror property.
- **U:** MatterSim/MACE rollouts, periodic solids, NVT/NPT, phase stability, and real throughput.
