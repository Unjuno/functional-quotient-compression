# MA-872 — discrete hypotheses in module-reuse structure space

This PA234 mechanism screen tests whether a new task drawn from a small set of learned module recombination rules can be identified as a discrete hypothesis from fewer examples than continuous task-code fitting.

**Mirror insertion:** this experiment adds a discrete Mirror address `m` at the module-recombination interface so that a new logical task selects one of eight shared rule prototypes without storing its full module coefficient vector.

Controls include native nearest-codebook search over the same prototypes, continuous rank-2 PCA code, random prototypes, independent coefficient fitting, and the teacher hypothesis. The central Mirror-specific control is native codebook search: if it selects the same code at the same bytes, the discrete Mirror address adds no distinct function.
