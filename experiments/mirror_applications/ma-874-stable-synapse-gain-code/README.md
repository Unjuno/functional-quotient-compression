# MA-874 — gain-modulated context with stable recurrent synapses

This PA235 mechanism screen tests whether a small dynamic context code can reconfigure a fixed recurrent reservoir to implement multiple temporal response functions, while retaining quality and reducing per-task writable state relative to fast-weight updates.

**Mirror insertion:** this experiment adds a two-scalar Mirror context `m` at the recurrent-state readout gain interface so that task-specific temporal filters can share the reservoir's fixed recurrent parameters.

Controls: native PCA/context-gain coordinates, independent task readout coefficients, pooled readout, and task-specific recurrent-decay fast weights. The native gain control is required to identify whether any benefit is simply a standard context/readout parameterization.
