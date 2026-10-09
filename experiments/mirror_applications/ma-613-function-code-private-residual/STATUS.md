# MA-613 status

Stage: deterministic mechanism/storage screen complete.
Decision: PROMISING, scoped to an analytic Fourier family.
H: Shared signature/phase codes plus sparse paid residuals can represent rare off-orbit functions more compactly than full per-function codes.
T: Two deterministic grids; private fractions 0, 12.5, 25, 50, and 100%; no/half/full residual recovery; full coefficient control; zero optimizer updates and no fresh split.
D: At 25% private, full residual recovery is exact to <2.5e-15 NRMSE2 at 2,969 B vs 5,401 B full (0.550x). At 100%, 3,353 B (0.621x). Half recovery at 25% has NRMSE2 about 0.01486.
C: The hand-designed Fourier basis, rather than a learned Mirror/interpreter, accounts for all observed compression.
U: Learned discovery, exception detection, natural tasks, adaptation noise, and independent fresh replication.
