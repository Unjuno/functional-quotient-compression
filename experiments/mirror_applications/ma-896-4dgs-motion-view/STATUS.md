# MA-896 status

**UNTESTED — blocked before training.** The official 4DGS implementation builds CUDA rasterizer and nearest-neighbor extensions and uses CUDA tensors/events in train/render paths. The available PyTorch is CPU-only, CUDA is unavailable, and no registered dynamic-scene assets were found. No candidate or control was trained or scored.

Resume with a compatible CUDA environment, built official extensions, and a fixed dynamic-scene dataset/config. Preserve the official native deformation and ordinary scene/motion-code deformation controls.
