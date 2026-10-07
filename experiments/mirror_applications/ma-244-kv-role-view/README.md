# MA-244 — K=V projection sharing + Mirror role recovery

Status: SCREENING
Evidence lane: MECHANISM / STORAGE / RUNTIME
Base commit: `5b736a0f7cfca9c3f7794005dfb70954f155ea02`

## H — hypothesis

A single physical MQA K/V projection plus small head-and-role Mirror coordinates can recover useful K-versus-V and per-head attention distinctions while storing fewer K/V projection parameters and cache states than ordinary MQA or MHA. It must beat a byte-near per-head gate control to justify Mirror-specific value.

## Prior art and exact delta

PA07 (*Do Transformers Need Three Projections? Systematic Study of QKV Variants*) reports Q != K=V as a strong sharing baseline, including 50% KV cache reduction with 3.1% perplexity degradation in its language experiments, and finds that projection sharing combines with GQA/MQA. MA-244 tests whether structured role addresses can recover role distinction after more aggressive K=V and MQA sharing. A tiny attention task is used as a mechanism screen, not as a replication of PA07.

## Controls

- MHA: independent per-head K and V projections (quality upper control).
- K=V MHA: separate key/value-head matrices are hard-tied per head.
- GQA-2: two K heads and two V heads shared over four query heads.
- MQA: one K and one V projection shared by all query heads.
- MQA K=V: one matrix and one cached vector serve both roles.
- Mirror MQA: one shared base projection, with one Givens angle for every query-head/role pair; one base projected cache is transformed on read.
- Gate MQA: byte-near scalar multiplier per head/role on the same one-projection base.

## Task and scope

The synthetic teacher uses one shared base projection with independently sampled role/head Givens transformations. That is an optimistic alignment screen for Mirror's stated coordinate family. Each sample is a one-step multihead attention query over an eight-token memory; target is teacher attention output MSE. The same query and memory inputs are used by all methods. This tests whether the mechanism can recover structured role asymmetry and quantifies actual bytes, cache-state size, active MAC proxy, and CPU runtime. No natural language or generalization claim follows.

## Gates

PASS requires, in all three fresh worlds: Mirror MSE within 10% of MHA, at least 20% lower projection payload than MQA, at least 10% lower MSE than hard MQA K=V, and within 10% MSE of or better than byte-near Gate MQA. Report cache bytes separately; a one-vector shared cache is charged as actually stored.

FAIL if the quality gate is missed or Gate MQA matches Mirror within 10% at equal or lower bytes. NOT ESTABLISHED if any fresh world or strong control is missing.

## H / T / D / C / U

H is stated above. T, D, C, and U will be written after the configuration and source are frozen and all fresh worlds are complete.
