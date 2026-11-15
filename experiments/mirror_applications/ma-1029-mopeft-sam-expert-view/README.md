# MA-1029 — SAM feature adapter-bank Mirror views

Status: SCREENING. The protocol is frozen in `PROTOCOL.json` before experiment data are generated.

This is a controlled feature-space screen of task-specific SAM adapter functions. A shared rank-8 bottleneck adapter receives per-domain Givens coordinates; comparisons include independent and hard-shared adapters, an equal-placement FiLM control, and a paper-inspired mixed Adapter/LoRA/prefix-gate control. The official MoPEFT repository could not be located in bounded source search, so this is not presented as an official MoPEFT or MESS reproduction.

All SAM modules are frozen. The trainable Adapter is placed on the final image embedding so updates remain feasible on CPU; every adapter/code byte and the frozen SAM deployment bundle are measured.
