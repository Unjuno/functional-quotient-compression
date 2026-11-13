# Execution blocker

MA-942 requires DCVC-UF native 64-QP rate-control modules, actual entropy-coded bitstreams and parallel decoder timing. MA-941 resource lookup found no official implementation/checkpoint/entropy coder/bitstream; this environment has no CUDA/NVIDIA accelerator. No equal-coded-rate control can be reproduced.

Disposition: **UNTESTED**, not FAIL. Revisit with the official 64-QP codec artifacts and compatible runtime.
