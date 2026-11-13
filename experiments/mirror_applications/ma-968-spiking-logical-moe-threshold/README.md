# MA-968 — One spiking expert with logical threshold specialists

Status: SCREENING. Protocol fixed before synthetic development/audit worlds.

## Hypothesis

One physical spiking synapse vector with small task-role threshold coordinates can replace independent expert synapses when tasks share synaptic structure; generic task-specific synapses show where private state becomes necessary.

## Mirror insertion

A single hard-threshold neuron uses a shared 16D synapse vector. Mirror codes generate a role-specific threshold as `base + scale * m_role`. Controls include ordinary direct threshold banks, rank-2 private synapse residuals, and independent experts with deterministic top-1 routing by the observed role.

## Prior-art delta

PA282 shows learned thresholds can alter SNN computation. MA-968 tests whether a small role code can express multiple useful threshold specialists on one physical synapse pool, compares the direct threshold-bank control, and measures a generic family where synapse sharing should fail.

## Results

See `RESULTS.md` after the fresh worlds and payload replay checks.
