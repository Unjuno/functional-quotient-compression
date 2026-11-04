# MA-671 — MoLE gates over Mirror-compressed logical LoRAs

Compare a full layerwise Mixture of LoRA Experts bank with a shared atom representation in which input-dependent gate weights are applied to expert codes before evaluating the shared atoms. This tests whether dynamic mixtures preserve the output of a dense expert bank with fewer stored bytes and active expert computations.

Native shared-basis MoLE and full-bank top-2 routing are required controls. Private rank-2 residuals measure the byte and compute cost of off-basis expert behavior. This is a synthetic two-layer screen, not a MoLE benchmark.
