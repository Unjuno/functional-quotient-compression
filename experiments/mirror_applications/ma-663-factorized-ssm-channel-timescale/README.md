# MA-663 — Factorized channel × timescale Mirror SSM

Test whether a task code over shared channel and timescale factors can store many domain-specific SSM transition tables. The synthetic teacher creates rate tables from a rank-4 channel × timescale tensor plus orthogonal per-task residuals. A stable softplus transform feeds each table into a diagonal recurrence.

Compare independent complete tables, ordinary CP codes, the same state called a Mirror coordinate, and CP plus private residual ranks. Measure sequence output quality, complete serialized bytes, transition generation compute and recurrent update compute. This is a controlled synthetic SSM screen, not a Mamba reproduction.
