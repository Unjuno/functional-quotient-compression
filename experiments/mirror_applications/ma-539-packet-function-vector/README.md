# MA-539 — Support-extracted function vector as a shared packet plan

Status: SCREENING. Evidence scope is a synthetic function execution screen.

## H — falsifiable hypothesis

Across two development worlds, a 16-dimensional function vector extracted from eight support input-output examples and shared across four packet slots will match an equal-width learned generic latent within 0.03 joint accuracy and 0.05 token NLL, use at least 10% fewer serialized inference bytes, and retain PTP-control valid-path rate.

## T — frozen protocol

PA99 motivates behavior-bearing activation vectors; PA10 and TM001/MA-248 motivate joint packet consistency. Each world contains 16 independent random permutations over 32 states. Eight support examples identify each function; held-out states form four-slot packets evaluated under the same function. Dev worlds: 53901/53902. Fresh worlds: 53911–53913, locked.

One shared two-layer width-32 MLP decoder predicts each slot. The candidate obtains a 16d function vector by averaging a shared support-pair encoder over eight demonstrations, then broadcasts that vector to all four slots. Controls are no function code, a directly optimized equal-width generic latent bank, four independent PTP-style slot-code inputs, and the explicit function table upper. All model, address, and metadata bytes are charged from actual uncompressed serialized inference payloads. The protocol selects a common development setting from updates {800,1200} and learning rates {.001,.003}; fresh opens only if every frozen gate passes on both worlds.

The complete protocol and SHA-256 freeze record were committed before code, world generation, or metrics.

## D — decision

Pending.

## C — strongest counter-hypothesis

The generic latent is already the cheapest function address; averaging support-pair activations may add encoder weights without reducing code or decoder state. PTP slot conditioning may achieve equivalent quality with less fragile optimization.

## U — boundaries

Synthetic permutations only; externally supplied support examples; one packet width and one decoder scale. No natural-language generation, learned retrieval, or production throughput claim.
