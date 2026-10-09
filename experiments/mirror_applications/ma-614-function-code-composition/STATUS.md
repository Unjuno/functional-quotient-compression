# MA-614 status

Stage: deterministic held-out composition screen complete.
H: composing small codes before one shared executor call could replace two sequential function calls.
T: two deterministic phase grids; 48 ordered pairs available and 16 held out per grid; exact sequential sine composition, a one-call ten-harmonic Mirror decoder, and explicit composite coefficient control; no optimizer updates.
D: FAIL. Sequential execution has zero error by construction. Mirror reconstruction has mean held-out NRMSE2 0.07624 and max 0.19942 in both grids. Its payload is 2,273 B vs 2,465 B sequential source state (-7.8%) and 2,977 B explicit composite codes, but active compute proxy is 65,696 vs 160 (~410x); measured CPU latency is about 2.8–2.9 ms vs 0.31 ms sequential. The predeclared <=1e-6 quality gate is missed.
C: The result is driven by the fixed ten-harmonic truncation and an unoptimized CPU implementation; a richer executor could reduce error but may further raise compute/state.
U: trained interpreter, learned code composition, nonlinear task quality beyond this analytic family, optimized kernels, GPU latency, and fresh replication. The mechanism screen is not a learned capacity study.
