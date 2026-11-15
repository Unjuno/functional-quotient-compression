# MA-1041 environment inspection

- `lsblk`: `vda disk ROTA=1 SIZE=32G`; `pmem0 disk ROTA=0 SIZE=14.3G` (persistent memory); no SSD block device reported.
- Root is overlay on `vda`; `/tmp` is tmpfs.
- System Python package probes: `diskannpy`, `faiss`, `scann`, `hnswlib` unavailable.
- Local `third_party/` scan found no DiskANN checkout.
- `g++` present; CMake/Ninja unavailable.
- No data, index, train/dev/audit split, model run, or query metric was produced.
