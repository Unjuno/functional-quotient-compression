# MA-539 — Packet-level function vector as shared plan

Status: SCREENING (protocol frozen; development not yet run)
Evidence lane: SYNTHETIC_PACKET_MECHANISM / ACTUAL_BYTES / COMPUTE
Base commit: `c2bac092`

## H — Hypothesis

A single compact function vector extracted from a support demonstration and shared across four future-token slots will improve joint packet consistency over independent phase-slot prediction and a generic shared packet latent, at no more than half the per-slot code bytes. The test is falsified if the function-vector path fails to beat both controls on held-out joint NLL and exact packet accuracy.

## T — Task and prior art

Use four-token packets in a deterministic rule-execution task whose rule is identified by a support demonstration. The demonstration is available before the packet and its extracted code must select the correct shared rule. This separates already-available function identity from hidden within-packet randomness (the TM001 failure boundary).

PA99 establishes causal function vectors in language models as an adjacent possibility, but does not show packet-level extraction or byte efficiency. PA10 establishes input-conditioned shared randomness for parallel dependent token prediction; compare a shared sampled packet latent. TM001 establishes a deterministic packet parallelism positive and a hidden branch failure for factorized slots; its packet-latent diagnostic was promising but below AR. This experiment is a synthetic mechanism screen, not natural language or capacity evidence.

## Mirror insertion

Native method: a shared rule table plus phase-specific slot decoder.
Mirror coordinate: one 8-dimensional extracted FV code shared by all four phase slots, selecting a logical rule from a common physical rule bank.
Cheapest controls: one shared 8-dimensional generic packet latent; four independent 8-dimensional per-slot codes; direct rule ID (upper bound, charged).

## Comparisons

1. Autoregressive executor with previous realized tokens (quality upper reference).
2. Independent phase slots with no shared plan.
3. PTP-style single packet latent shared across all slots.
4. Generic learned packet latent, same dimension and extraction budget as FV.
5. Support-extracted FV shared by every slot.
6. Explicit rule ID shared by every slot (oracle ceiling; bytes charged).

## Frozen protocol

- Development worlds: 60101, 60102. Fresh worlds: 60111, 60112, 60113; sealed until development gates pass.
- P=4; 16 symbols; 8 bijective rules per world; 64 support demonstrations and 256 query packets/world. Demonstrations are disjoint from queries. Each support contains an input/output pair; FV extraction receives only the support pair, not the query target.
- Each packet applies the selected rule four times to the start symbol; all slots depend on the preceding realized symbol. Every condition uses the same exact finite-state executor; no model optimizer or training updates are used.
- Amendment 1 replaces optimization with an exact finite state extraction/execution audit (zero updates): each logical function is supplied as its own support set, and a one hot function address is recovered by support consistency against the shared rule bank. Query targets are not shown to the extractor. No hyperparameter search is performed.
- Primary: exact packet accuracy and valid path; secondary: serialized address and common-base bytes, plus a common exact-executor CPU calibration. Joint NLL is not estimated by this exact deterministic screen.
- Storage: actual deterministic serialized inference payload bytes, including rule bank, FV encoder/code, latent parameters, slot heads, metadata and models. Count each method's incremental bytes plus common executor once. Direct rule ID is a charged oracle upper bound.
- Compute: training examples/updates, analytic active multiply-add count per packet, measured isolated CPU wall time; one thread. No capacity claim from code combinations.

## Gates

PASS screen: FV exact packet accuracy is >=95% in each development world, and FV address bytes are <=50% of four independent slot codes. Then open frozen fresh worlds to replicate these two conditions. FV-specific value additionally requires a measurable advantage over the equal-byte generic shared latent; equality fails the FV-specific claim even if shared-address compression succeeds.

FAIL: either development world misses NLL/accuracy gate, or FV is no smaller than the generic latent at matched function quality. Fresh remains sealed on development failure.

NOT ESTABLISHED: implementation/control mismatch, failed deterministic reconstruction, or task leakage.

## C / U

Strong counter-hypothesis: deterministic rule identity is already a compact categorical variable, so generic shared latent or direct rule ID explains any gain; the learned FV adds no useful compression. Unconfirmed: transfer to natural-language function vectors, unseen rules, stochastic packets, and production decoding.

Facts, interpretations, and hypotheses will remain distinct in the final report.

## Results and decision

**FACT:** Support extraction recovered every rule address and all FV outputs were exact in development worlds 60101/60102 and fresh worlds 60111/60112/60113. Each world had 256 query packets; the minimum support examples per rule ranged from 3 to 6. The common rule bank plus its inference config serialized to 1,207 B. The shared 8-rule address including its canonical inference config was 468 B; four repeated slot tables plus config were 1,245 B (37.6%). The full finite-state payload was 1,675 B versus 2,452 B after charging common state once. Every payload hash is recorded under `runs/`.

**FACT:** FV, PTP shared latent, generic shared latent, and direct rule-ID oracle use identical shared-address tensors, canonical configs, serialized bytes (468 B), and SHA-256 hashes within each world; all achieve 100% exact packet accuracy. The common executor calibration over 1,000 passes ranged from 0.1524 to 0.1746 s across runs and does not separate methods. Training examples: 64 support pairs/world; optimizer updates: 0. Joint NLL is not estimated by this exact deterministic screen.

**INTERPRETATION:** Sharing one function address across four positions cuts repeated address bytes, but the generic-latent equality shows this is ordinary address sharing. The task is too easy to establish an FV-specific benefit.

**H/T/D:** MA-539 **FAIL** for FV-specific value. The synthetic physical-address factorization works; independent capacity is not established.

**C:** Each support set belongs to a known rule in a fixed shared bank, so support consistency reduces to categorical rule-ID recovery. A generic latent or direct ID explains the result.

**U:** Learned language-model FVs, unseen-function transfer, stochastic packet uncertainty, neural joint NLL, and method-specific inference runtime remain untested.

**Protocol deviation:** amendment-1 was authored before any measurement, but its amended protocol/code were not committed before the first development run. After fresh access, accounting was completed for the canonical JSON inference configs required by the storage contract; this changed serialized byte totals but no model behavior or selection. All run data and deviations are retained; fresh worlds were first opened only after the amended development byte/accuracy gate passed.
