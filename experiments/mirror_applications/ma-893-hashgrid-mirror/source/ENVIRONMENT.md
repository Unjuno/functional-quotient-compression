# Environment feasibility probe

Observed in the current container:

- PyTorch: `2.14.1+cpu`
- CUDA available: `false`
- PyTorch CUDA build: `none`
- `nvidia-smi`: unavailable
- `tinycudann`, `torch_ngp`, `nerfacc`: not importable

The registered Instant-NGP fused CUDA baseline cannot be run in this environment.
