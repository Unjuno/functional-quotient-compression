# MA-707 replay/stability amendment

The initial run reported the spectral radius of the unscaled orthogonal factor, omitting the registered 0.95 contraction, and compared a trajectory array with itself rather than replaying from serialized candidate state against the independent stored transition. Initial source and results are preserved under `source/attempts/initial_stability_and_replay/` and are invalid for stability/replay conclusions.

The corrected run applies the frozen scale to the spectral-radius calculation and reconstructs candidate transitions from serialized Q0/U/V payloads before comparing trajectories with independent Q matrices. No targets, seeds, ranks, thresholds, or tuning choices changed. The same locked audit seeds are replayed as a verification correction.
