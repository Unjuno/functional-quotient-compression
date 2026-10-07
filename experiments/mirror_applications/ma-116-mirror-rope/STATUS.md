# MA-116 status

- Status: FAIL at development gate
- Branch: `research/ma-116-mirror-rope-20261007`
- Base commit: `68775c5784f0954adcf72147535a8569adfd8bd9`
- Last verified commit: pending
- Development complete: yes; seeds 11601–11602
- Fresh/audit opened: no; independent-frequency quality and byte gates failed
- Results committed: pending
- Verification committed: pending
- Registry row updated: pending

## H / T / D / C / U

- **H:** A compact Mirror frequency coordinate can generate domain-specific RoPE schedules, extrapolate long positions, and save actual serialized bytes over independent schedules while beating scalar scaling.
- **T:** Four synthetic rotary domains, four frequency bands, train positions 0–15 and evaluate 16–127. Compared fixed RoPE, learned scalar scaling, Mirror address-direction, and independent frequencies. Two development seeds, 800 Adam updates at LR 0.02 for trainable methods.
- **D:** FAIL: Mirror strongly beat scalar scaling on the aligned long-position task, but independent frequencies were nearly exact and serialized smaller (2,021B vs 2,209B). Fresh stayed sealed.
- **C:** The teacher follows Mirror's address-direction factorization by construction, but the separate address and direction tensors add enough serializer metadata that independent tables still have a better actual-byte/quality frontier.
- **U:** Packed production serialization, attention quality, natural-language NLL, perplexity, cache effects, larger frequency counts, and model-level transfer.

**FACT:** 8/8 dev rows replayed; exact payload bytes; max metric delta 4.16e-11; tests 2/2 passed. Mirror median held-out MSE 8.32e-4; independent frequencies 2.25e-13; scalar scaling 0.9895. Mirror payload 2,209B vs independent 2,021B.

**INTERPRETATION:** A learned shared frequency direction helps the structured teacher extrapolate, but does not improve the measured storage/quality Pareto frontier under the current actual serializer.

**HYPOTHESIS:** More frequency bands or packed tensor serialization could change the storage comparison; neither is established here.

## Next action

Update the registry and program status, commit and push this development failure, then continue to MA-121.

## Blockers

None.

## Decisions / rulings

Fresh worlds 11611–11613 remained sealed after the development failure gate. Fixed RoPE has no optimizer state and is reported with zero updates/examples.
