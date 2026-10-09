# MA-357 — Hopfield reservoir for Mirror addresses

Status: **FAIL at development; fresh sealed**  
Branch: `research/ma-357-hopfield-mirror-address-reservoir-20261009`
Base: `c935a90`  
Prior art: PA44 modern Hopfield networks; PA31 explicit code/mask banks.

## H — Hypothesis

Storing logical-view addresses as associative attractors can recover corrupted Mirror addresses with lower serialized address bytes than an explicit packed code table, while preserving selected function quality and limiting interference as the number of stored views grows.

## Mirror insertion

> **Mirror insertion:** this experiment adds a Hopfield address retrieval layer before the shared function decoder so noisy compact queries can recover the intended logical Mirror code without storing a separate dense router.

- Shared object: one synthetic 16→8 linear-view decoder basis, shared by 64 task views.
- Mirror state: each view has one 8D logical function coordinate, addressed by a 32-bit binary attractor.
- Native controls: explicit packed address table with direct lookup; nearest-neighbor Hamming lookup; softmax modern Hopfield retrieval.
- All addresses, function coordinates, temperatures, indices and metadata count toward payload bytes.

## T — Frozen protocol

Three development seeds 35701/35702/35703; fresh seeds 35711/35712/35713 remain sealed unless all development gates pass. Store K∈{16,64,256} random binary 32-bit attractor addresses and 8D Gaussian view codes. Generate 2,000 random queries per K by flipping 0–8 bits uniformly. Compare packed explicit exact lookup, Hamming nearest address, and modern Hopfield softmax attention over the stored attractors. Decode retrieved function codes through the same fixed shared basis and measure code retrieval accuracy, decoded-view nMSE, collisions, bytes, and query dot-product MACs.

## Gates

PASS requires for K=64 and 256: at least 99% exact view recovery at <=2 flipped bits, decoded nMSE <=1e-4, >=10% fewer total address bytes than packed explicit table, and query MACs <= explicit lookup. FAIL if storage savings <10%, Hamming control matches retrieval, quality fails, or Hopfield compute >1.1x Hamming retrieval.

## Boundaries

Synthetic random codes and fixed decoder only; no trained attractor learning, natural data, neural-network capacity, or end-to-end routing. Attractor count is not capacity. Report interference as a function of K and corruption level.

## Result

**FAIL.** On all three development seeds, packed explicit Hamming and Hopfield used the same address/code state bytes (method metadata accounts for a few bytes), but Hamming had equal or better retrieval. At K=64, Hamming accuracy was 0.976–0.985; Hopfield matched the argmax accuracy but soft mixtures raised decoded-code nMSE to 0.183–0.200. At K=256, accuracy fell to 0.925–0.941 and Hopfield decoded-code nMSE was 0.416–0.426. Hopfield query MACs were 25% higher than Hamming; no address-byte reduction was observed. Fresh seeds remained sealed.

**Fact:** Retrieval errors/interference rose with bank size; modern Hopfield code averaging harmed function reconstruction relative to hard Hamming retrieval.

**Interpretation:** Associative retrieval over binary addresses did not compress the packed code table and introduced interference under corruption. A hard nearest-neighbor control was stronger on function quality at lower compute.

**C:** Random binary attractors and uniform 0–8 bit corruption may not represent learned task-code neighborhoods.

**U:** No learned attractor dynamics, semantic code geometry, natural tasks, or trained shared decoder was tested.
