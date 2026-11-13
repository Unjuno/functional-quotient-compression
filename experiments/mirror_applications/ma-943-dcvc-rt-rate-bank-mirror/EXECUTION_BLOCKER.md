# Execution blocker

- PA270’s real-time neural video codec is a hard 1080p FPS/kernel-call/memory-I/O test.
- GitHub search returned one third-party implementation; its README states that pretrained-weight inference did not work due to dependencies. The checkout includes no `.pt/.pth/.bin` checkpoint files or benchmark datasets.
- This environment has no CUDA/NVIDIA GPU.
- No official DCVC-RT native rate bank, reference bitstream or 1080p benchmark assets are available.

Disposition: **UNTESTED**, not FAIL. Do not treat the third-party module-bank code as reproduction of native DCVC-RT.
