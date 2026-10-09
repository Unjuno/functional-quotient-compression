# MA-557 — MatFormer FFN granularity and Mirror neuron roles

Status: **FAIL for Mirror-specific value**  
Branch: `research/ma-557-matformer-neuron-roles-20261009`  
Base commit: `7ede5c3e`  
Prior art: PA107 MatFormer

## H

Per-granularity neuron role codes recover distinct width-specific functions from shared nested FFN neurons with fewer bytes than independent widths; the question is whether the role representation adds value beyond ordinary channel gains.

## Frozen screen

An 8-neuron physical FFN supports nested widths 2, 4, and 8. Three target operators are generated from the shared physical basis with independent diagonal channel roles. Compare plain nested prefix, Mirror role gains, direct ordinary gains, independent width-specific FFNs, and a native-tail model storing width-specific tail neurons. Two development seeds; 512 Gaussian probes/width; exact outputs and deterministic NPZ payloads are measured. All bases, gains, indices and metadata are charged. This is an aligned mechanism/storage screen, no training updates.

PASS requires all widths nMSE ≤1e-6, ≥20% fewer bytes than independent width FFNs, and ≥10% fewer bytes than direct gains at matched quality. FAIL if direct gains match Mirror or plain nested prefix already meets quality.

## H / T / D / C / U

- **H:** Mirror neuron roles recover useful width-specific behavior with shared physical neurons and compact codes.
- **T:** Widths 2/4/8; two dev seeds; five controls; 30 rows, no optimizer updates.
- **D:** FAIL for Mirror-specific value. Mirror/direct gains both 1016B and exact at nMSE 0; independent widths 1249B. Plain nested prefix 739B but mean nMSE .06485.
- **C:** MatFormer nested FFNs and ordinary diagonal channel gains may fully explain the result.
- **U:** Actual MatFormer pretraining, language NLL, learned role codes, GPU throughput.
