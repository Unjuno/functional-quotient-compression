# MA-971 — LightPro physical coupler bank

Status: NOT ESTABLISHED / execution blocked.

The registered claim requires comparing sparse Mirror coupler settings against fully programmed LightPro hardware, including optical quality, programmed bits, switch energy and footprint. The current container has no photonic instrument/API, LightPro device access, or optical simulator. CPU matrix emulation cannot measure the registered physical metrics and is not substituted.

## H / T / D / C / U

- **H (hypothesis):** a sparse Mirror task code can realize multiple useful operators on one physical phase-change coupler bank with lower setting/storage and switching cost than full reprogramming.
- **T (fact):** PA286 describes a programmable phase-change directional coupler processor; local inspection found no device SDK or data and no photonics packages (`meep`, `tidy3d`, `lumapi`, `photontorch`). No GPU is present either.
- **D:** NOT ESTABLISHED; registered physical hypothesis could not be tested.
- **C (interpretation):** all apparent setting-count savings in an ideal matrix model could be erased by calibration, optical loss, thermal switching, or controller overhead.
- **U:** real LightPro hardware/API, matched full-programming control, calibrated task workloads, and measured switch energy/latency/optical loss.

No Mirror result or quality/storage frontier is claimed.
