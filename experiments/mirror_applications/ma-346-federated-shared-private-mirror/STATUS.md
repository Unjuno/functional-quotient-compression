# MA-346 Status

**PROMISING only for the synthetic shared/private storage frontier at 25–50% heterogeneity.** On all three fresh seeds, phase plus rank-1 residuals retained exact quality and used 6,739B/8,639B versus 20,243B/35,696B FedRep dense-private (66.7–75.8% fewer bytes). Client communication was 2,336B/4,416B versus 16,928B/33,344B. No-private outlier error exceeded the 1e-5 gate. The native scalar-phase control was byte/hash identical; no Mirror-specific advantage.

Three tests pass and 175 payload/hash/metric rows replay exactly. See README for H/T/D/C/U and fact/interpretation/hypothesis.

Integrated from dedicated branch `research/ma-346-federated-shared-private-mirror-20261008` (commit `bf8030d64cdac7abcf855569bbad31ba27e7d9ad`); 175 payloads replay exactly and three tests pass. Native scalar-phase control is byte/hash identical.
