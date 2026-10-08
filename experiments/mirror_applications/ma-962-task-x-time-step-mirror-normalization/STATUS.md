# MA-962 status

- Status: **FAIL** (development Mirror-specific gate; audit unopened)
- Branch: `research/ma-962-task-time-mirror-tebn-20261008`
- Base commit: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`
- Draw 15 replay: passed; pool size 553, index 405, seed and pool stored
- Protocol: frozen and amended before any MNIST access
- Development: ten runs completed (five conditions × two seeds)
- Quality/storage gate: passed in both seeds; payload ratio 0.9716× native TEBN
- Mirror-specific gate: failed against equal-byte rank-1 task-scalar control in both seeds
- Audit: official test files not downloaded or opened
- Total training wall time: 117.6 s, single-thread CPU
- Payload verification: 10/10 hash/size/strict-reload checks passed

## H / T / D / C / U

**H:** A shared temporal gain profile plus one task phase coordinate can preserve TEBN quality at >=2% fewer complete payload bytes and beat the equal-size scalar gate by >=1 point.

**T:** Four rate-coded MNIST temporal tasks; one 784/128/10 LIF SNN; five conditions; two development seeds; 800 updates per shared condition and per independent task; audit gated on both development checks.

**D:** **FAIL.** The Mirror passed quality/storage (414,436 B vs 426,532 B TEBN, 0.9716×) but underperformed the rank-1 task-scalar control by 0.293 and 0.208 points. Audit remained unopened.

**C:** A scalar task gain explains the task variation as well as or better than phase-shifted TEBN.

**U:** Official test generalization, other temporal tasks, capacity, and hardware latency/energy remain unknown.
