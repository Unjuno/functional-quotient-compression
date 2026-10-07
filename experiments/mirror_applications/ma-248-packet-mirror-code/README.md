# MA-248 — PTP random variable represented as packet Mirror code

Status: SCREENING  
Evidence lane: MECHANISM / STORAGE / RUNTIME  
Base commit: `f91625f2fe1b110593b16605c26fc9c7675c1824`

## Hypothesis

A shared categorical packet address with phase-specific low-description Mirror views can preserve jointly consistent P=4 outputs when branch uncertainty is correlated. With independent per-position uncertainty, address entropy should grow from one to four bits and match PTP. Any quality or storage gain must beat a simple broadcast-code control, not just exploit the smaller correlated source.

## Prior art delta

PA10 (Parallel Token Prediction, ICLR 2026) feeds one random auxiliary `u_i` per future position. Each future token is a deterministic function of context and its own plus preceding auxiliaries; this construction can represent arbitrary dependencies. TM001 found that factorized period slots failed when a single hidden branch was shared across the packet; a small packet latent helped but did not close the gap. MA-248 tests a narrower proposition: whether one packet address, read through phase-specific Givens views over a shared decoder, can encode correlated packet randomness compactly, and where it stops helping as branch entropy becomes independent across phases.

## Experiment design

The teacher predicts four state transitions from 16 states and 8 rules using two random branch-specific permutation tables. In the correlated condition one binary branch is sampled once per packet, giving two possible packet codes. In the independent condition each of four phases draws its own branch, giving sixteen possible packets. Inputs contain only context plus the method's registered random address; target tokens are never fed to future slots. Methods: PTP-style factorized branch auxiliaries, no-aux direct slots, shared packet embedding, scalar-gated shared packet embedding, Givens Mirror packet views, and untied phase-specific packet embeddings.

## Predeclared gates

See `PROTOCOL.json`. The key measurements are exact joint packet accuracy, token NLL, valid path, serialized model bytes, random-address bits per packet, combined bytes, compute proxy and CPU wall time. Random address cost is kept separate from model payload to show which frontier improves.

## Results

Do not write conclusions until development selection, source freeze, fresh evaluation, and metric replay are complete.

## Decision

FACT: pending.  
INTERPRETATION: pending.  
HYPOTHESIS: pending.  
BOUNDARY: pending.
