# MA-311 status

**FAIL** — preregistered promotion gate requires <=0.85x the ordinary intrinsic payload and all aligned tasks represented as views in every fresh world. World 31123 measured 0.859x and used one private fallback for an aligned task.

The result retains a scoped positive storage observation (14.1–21.1% savings in 3/3 fresh worlds) and a quality tradeoff (Mirror MSE around 5.8e-6–1.9e-5 vs ordinary intrinsic around 1.6e-8–1.9e-8). This FAIL does not imply no representational value; it means the compound preregistered gate was not met.

## H / T / D / C / U

- **H:** shared intrinsic vector + compact Givens coordinate should recover aligned functions at <=0.85x ordinary intrinsic bytes; unrelated functions fall back to private vectors.
- **T:** 128-D synthetic linear tasks, d=64 random projection, 8 tasks, tied/intrinsic/Mirror/independent controls; dev 31120–31121; locked fresh 31122–31124.
- **D:** FAIL for strict all-world promotion.
- **C:** tasks were generated from the same Givens orbit as the method, and ordinary intrinsic vectors still had substantially lower error.
- **U:** natural task distribution, pretrained models, optimization, capacity and deployment latency.

## Fact / interpretation / hypothesis

See `README.md`; all three are separated there. Artifact/result replay is exact for the checked metrics and payload lengths.
