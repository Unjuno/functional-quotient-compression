# Upstream and resource provenance

- Official HNeRV PyTorch: https://github.com/haochen-rye/HNeRV
- Pinned commit: `4872129c8d004a25477e0c1ffbbff4ba71943ad5`.
- Included `data/bunny`: 132 PNG frames; only one scene/video supplied.
- `checkpoints/img_decoder.pth`: 6,005,995 B; `checkpoints/quant_vid.pth`: quantized per-video embedding/model state. Exact byte total measured was 7,574,217 B.
- CPU smoke used the repository’s `dequant_tensor` arithmetic and TorchScript decoder; input embedding shape `(132,16,2,4)`, one decoded output `(1,3,640,1280)`, all finite.
- NerVast PA268 source/checkpoints/Fisher masks are not present in this HNeRV repository or workspace.
