# MA-900 source and environment review

## Evidence collected (2026-11-13)

- PA265/arXiv 2607.24403 v1 identifies the work as GenSplatCodec and describes a dual stream (compact Gaussian structure plus reference appearance), geometry-guided one-step diffusion decoding, and three-stage codec/decoder optimization.
- arXiv API metadata has no `comment` field / code URL. Searches of GitHub repositories for `GenSplatCodec`, and public indexed code/documentation for the exact title, method, and official implementation, did not identify an official repository. This bounded search is not proof of nonexistence.
- Runtime has PyTorch 2.14.1+cpu; `diffusers` and `gsplat` are absent; `/dev/nvidia0`, `/proc/driver/nvidia`, and `/usr/local/cuda` are absent. No matching Gaussian scene/dataset directories were found in the workspace.
- A public search result surfaced ExGS, a different paper and codec. It is not GenSplatCodec and cannot substitute as the registered native control.

## Decision boundary

MA-900 requires the native structure/appearance codec and one-step generative decoder, actual encoded stream size, multi-view quality consistency, and latency/NFE-matched controls. Without official source, trained decoder/checkpoints, scenes, CUDA, and renderer, a toy diffusion decoder or other Gaussian codec would not be faithful. No model run, bytes, quality, or latency were measured. Keep UNTESTED, not FAIL.

## Sources

- https://arxiv.org/abs/2607.24403
- https://export.arxiv.org/api/query?id_list=2607.24403
- https://arxiv.org/pdf/2607.24403v1
- GitHub repository search `GenSplatCodec` and public developer search of title/code (2026-11-13).
