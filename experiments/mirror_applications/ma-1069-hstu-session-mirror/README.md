# MA-1069 — HSTU session-context Mirror views

Status: NOT ESTABLISHED (environment blocker; registry remains UNTESTED)  
Evidence lane: QUALITY / STORAGE / RUNTIME  
Base commit: `16807a7d6dc17a834a44ed3cc793ccc28b615691`  
Random draw: #15, eligible pool 536, index 56, selected MA-1069.

## Hypothesis

H: A small session-context coordinate `m` on a shared HSTU readout yields useful multiobjective user-preference functions without duplicating HSTU parameters or interaction-history state, at matched recommendation quality and serving cost.

## Mirror insertion

> **Mirror insertion:** this experiment adds `m` to the native HSTU readout so that session-conditioned user-preference functions can vary without duplicating HSTU weights or history state.

- Native method: HSTU sequential recommendation with historical conditioning (PA342).
- Proposed interface: compact `m` supplied to the readout.
- `m` persists per logical preference function; history/cache is separately charged.
- Cheapest ordinary control: session embedding, gate or low-rank readout modulation.

## Prior-art delta and controls

PA342 establishes HSTU as a sequential transducer for generative recommendation. This experiment asks only whether an additional compact readout coordinate adds useful function per byte/compute beyond native history conditioning. Required controls are native HSTU, shared HSTU plus ordinary session embedding/gate/low-rank readout, and independent heads where practical.

## Frozen gates

PASS requires fresh chronological recommendation results showing useful objective-specific quality at matched or better total serialized bytes and serving latency versus native HSTU and the cheapest ordinary control. FAIL if the ordinary control matches at equal or lower cost, or if required private state erases the benefit. NOT ESTABLISHED applies when native HSTU cannot be run.

## Execution record

FACT: The workspace lacks PyTorch, TensorFlow, JAX, Transformers, Datasets, CUDA tools and a GPU. No native HSTU implementation/checkpoint or recommendation data is available here. No development or fresh data was opened; no model was trained or benchmarked.

INTERPRETATION: The registered native-model comparison cannot be judged in this environment. A toy sequence substitute would not test HSTU or satisfy PA342 controls.

HYPOTHESIS: Whether session-context `m` adds useful HSTU functions remains unresolved.

## Unconfirmed

HSTU reproduction, temporal splits, ranking quality (NDCG/recall/AUC), complete serialized state including history cache, and serving P99 are all unmeasured. The experiment remains NOT ESTABLISHED; registry scientific status stays UNTESTED.
