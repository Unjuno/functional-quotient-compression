# MA-492 — Quantized packet-plan Mirror latent

Status: **FAIL** for the preregistered K<=8 coverage gate. At K16, discrete packet plans show a useful scoped storage tradeoff.

## H / T / D / C / U

**H:** A compact discrete plan code can preserve held-out joint future modes at lower serialized bytes than independent per-token distributions.

**T:** Synthetic 256-context two-step packet task, 16 legal joint modes over an 8-symbol vocabulary. Compare shared VQ plan codebooks K={2,4,8,16}, continuous context probabilities, independent categorical marginals, and greedy baseline. Development selected K16 as the max coverage setting. A1 fresh worlds 49220-49222 × seeds 0-2; A0 invalid-mode/coverage results are preserved as exploratory and excluded.

**D:** FAIL under the preregistered gate. K<=8 reached only 12.5%, 25%, and 50% joint-mode coverage, below 95%, despite 100% valid packet rate for emitted plans. K16 reached 100% coverage and valid rate at 3,173B mean, versus continuous 18,469B and independent categorical 18,213B; independent packet validity was 59.5%. K16 joint NLL .693 matched continuous, while K8 NLL was 10.662. The K16 result is a promising scoped storage/validity tradeoff but does not meet the registered small-code success criterion.

**C:** This task explicitly encodes 16 legal joint modes; requiring >=95% coverage with <=8 single codewords is structurally infeasible. Codebook overhead is small because the decoder and token vocabulary are synthetic and fixed.

**U:** Learned packet plans, realistic autoregressive contexts, language-model perplexity, and throughput on a deployed decoder remain untested.
