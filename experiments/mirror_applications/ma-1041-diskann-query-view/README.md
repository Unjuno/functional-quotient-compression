# MA-1041 — DiskANN graph traversal with query-state View

Status: UNTESTED (execution blocked; no model/index metrics)  
Evidence lane: ANN / SSD RUNTIME / QUALITY / STORAGE  
Prior art: PA324, DiskANN SSD graph retrieval

## Registered hypothesis

A small query/task View can specialize traversal of one shared SSD-resident DiskANN graph, preserving useful recall while avoiding graph duplication and improving the recall / SSD I/O / p99 tradeoff over native `search_list_size` tuning and a simple query-conditioned traversal gate.

> **Mirror insertion:** this experiment would add `m` to the query/traversal policy of one physical SSD graph so multiple task-specific search policies could share its graph topology.

## Execution blocker

The current container exposes a root filesystem on virtual `vda` reported as rotational (`ROTA=1`), and `/tmp` is tmpfs. There is no directly available SSD device on which the registered random-read and tail-latency claim can be measured. The runtime also lacks DiskANN/DiskANNpy, CMake, and a local official DiskANN source checkout. A Python ANN substitute or an in-memory graph would not establish the registered DiskANN SSD I/O/p99 hypothesis.

No dataset, index, audit split, quality metric, SSD read, or latency measurement was produced. Registry status remains **UNTESTED**, not FAIL. Evidence and resume requirements are in `EXECUTION_BLOCKER.md` and `source/ENVIRONMENT.md`.

## Resume requirements

A host exposing a directly testable SSD plus the pinned official DiskANN implementation and build dependencies. Before evaluation, freeze a new task/query split and compare native search-list-size tuning, query-conditioned simple gate, Mirror traversal View and exact retrieval; charge graph files, vectors, ID maps, metadata and every query-state byte.
