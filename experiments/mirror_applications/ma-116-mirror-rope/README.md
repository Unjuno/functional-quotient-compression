# MA-116 — Mirror-RoPE across positional domains

Status: FAIL at development gate
Evidence lane: MECHANISM
Base commit: `68775c5784f0954adcf72147535a8569adfd8bd9`

## Hypothesis

H: A shared RoPE frequency basis plus one learned scalar per domain and a shared per-frequency direction can generate several positional schedules, extrapolate long positions, and beat scalar scaling while using fewer serialized bytes than independent frequency tables.

## Physical-to-logical claim

- Physical object: one shared four-frequency RoPE base.
- Mirror coordinate: four learned domain addresses plus a shared learned direction over frequency bands.
- Logical multiplicity: four domain-specific frequency schedules.
- Failure modes: scalar scaling may be simpler; direction state and tensor metadata may cost more than independent schedules.

## Prior-art delta

PA13 proposes attention interpreted as role/filler binding. It is adjacent, not direct evidence for RoPE frequency sharing. This screen compares standard fixed RoPE, learned scalar frequency scaling, Mirror's shared frequency direction, and independent per-domain frequency tables. No novelty claim is made from this prior-art map.

## Development results

Two development worlds; train relative positions 0–15 and evaluate held-out positions 16–127. Fixed RoPE has no learned parameters and therefore uses zero optimizer updates.

| Method | Median train phase MSE | Median held-out phase MSE | Serialized bytes |
|---|---:|---:|---:|
| fixed RoPE | 0.1846 | 0.8346 | 1,705B |
| learned scalar scaled RoPE | 0.0711 | 0.9895 | 1,957B |
| Mirror-RoPE | 1.05e-5 | 8.32e-4 | 2,209B |
| independent per-domain frequencies | 2.97e-15 | 2.25e-13 | 2,021B |

Mirror improved held-out error by about 1,190× relative to scalar scaling, but independent frequencies were nearly exact and had a smaller actual serialized payload. Mirror has eight learned scalars, half the independent frequency count, but `torch.save` serialized its two learned tensors plus metadata larger than the independent single frequency tensor. The actual byte metric therefore rejects the compression claim in this payload format.

## Decision

**FACT:** Both development seeds show the same ordering. Mirror held-out MSE median was 8.32e-4 at 2,209B; independent per-domain frequencies reached 2.25e-13 at 2,021B. Scalar scaling had 0.9895 MSE at 1,957B. The preregistered independent-quality and byte gates failed, so fresh seeds remained sealed. Eight rows replayed with exact serialized bytes and maximum metric delta 4.16e-11; tests passed 2/2.

**INTERPRETATION:** Mirror's structured frequency direction helps extrapolate the deliberately address-structured teacher beyond short training positions, while independent frequencies remain more accurate and smaller under the measured serializer. Tensor count and metadata overhead dominate this tiny model payload.

**HYPOTHESIS:** A packed deployment format or larger rotary basis might move the storage frontier, but that is a separate experiment and cannot revise this result.

**BOUNDARY:** Synthetic sin/cos phase reconstruction only; no attention outputs, language modeling, perplexity, cache behavior, or optimized inference claims.
