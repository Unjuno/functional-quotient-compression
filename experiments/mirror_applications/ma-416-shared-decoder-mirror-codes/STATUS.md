# MA-416 status: FAIL

## H — falsifiable hypothesis

Two per-function Givens angles in a shared decoder would approach a two-dimensional ordinary latent code's held-out query reconstruction with materially smaller actual bytes.

## T — executed

16 random continuous Fourier functions (frequencies 1, 2, 3) per world; 64 train query samples per function; test on 128 held-out query points per function. Compared shared-only, Mirror2, ordinary latent1/2/8, and independent decoders. 500 AdamW updates × batch 256; 2 development worlds × 3 seeds selected LR 0.01 for all methods; 3 fresh worlds × 3 seeds. Actual serialized decoder and all codes retained. CPU PyTorch 2.14.1.

## D — FAIL

Fresh mean normalized RMSE: shared 0.9904; Mirror2 0.8884; latent1 0.8037; latent2 0.7264; latent8 0.5151; independent 0.5173. Payload bytes: Mirror2 2,905B vs latent2 3,161B (8.1% smaller), but Mirror error is 22.3% higher than latent2 and therefore misses the <=10% quality gate. Latent1 also has better quality with only 2.2% more bytes. Mirror MAC proxy is 72/query vs latent2 128/query. The reduced compute/bytes do not compensate for reconstruction loss in the registered gate.

## C — strongest counter-hypothesis

Random Fourier function variation is not well represented by rotating hidden feature pairs; ordinary input latents provide more flexible conditioning. Independent decoders and latent8 achieve similar quality but with much larger payloads, reflecting this short fixed-update setup.

## U — unresolved

Structured phase/amplitude function families, more code dimensions or richer Mirror coordinates, 3D signed-distance reconstruction, longer convergence, and inference throughput remain untested. This small 1D screen is not DeepSDF evidence.

