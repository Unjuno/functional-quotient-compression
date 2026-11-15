# MA-1041 execution blocker

## Fact

- Registry hypothesis requires one SSD-resident DiskANN graph and tests the recall, SSD reads, QPS and P99 latency tradeoff (PA324).
- `lsblk -o NAME,TYPE,ROTA,SIZE,MOUNTPOINT,FSTYPE` reports `vda disk ROTA=1 SIZE=32G`; `pmem0` is a 14.3G persistent-memory device, not an SSD block device. The workspace root is an overlay backed by `vda`; `/tmp` is tmpfs.
- System Python probes find no `diskannpy`, `faiss`, `scann` or `hnswlib`; no local DiskANN checkout exists under `third_party/`.
- `g++` exists, but CMake and Ninja are absent. No SSD-specific DiskANN index, data split or query was run.

## Interpretation

The required native SSD traversal baseline and physical I/O/P99 comparison cannot be reproduced on the available storage/runtime. An in-memory graph or another ANN library would answer a different question and is not substituted.

## Hypothesis

The DiskANN query-View hypothesis remains **UNTESTED**. This is an execution blocker, not negative model evidence.

## Requirements to resume

- A host/container with a directly exposed SSD block device and repeatable physical SSD I/O measurements.
- Pinned official DiskANN source and compatible build/runtime dependencies.
- A new fresh query set and benchmark data not used to tune traversal parameters.
- Freeze native DiskANN parameters, matched query-gate and Mirror controls before audit; measure recall@k, exact-search overlap, SSD/page reads, index bytes, build cost, QPS and P99.
