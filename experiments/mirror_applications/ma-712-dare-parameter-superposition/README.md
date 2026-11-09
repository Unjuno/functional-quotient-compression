# MA-712 — DARE before parameter-superposition unbinding

Status: SCREENING  
Evidence lane: MECHANISM/STORAGE/QUALITY  
Base commit: `c3521fb2`

## Hypothesis

H: DARE sparsification then context binding into one delta tensor can preserve per-task function quality at lower bytes than independent DARE deltas; Mirror contexts must beat standard Parameter Superposition controls.

## Mirror insertion

> **Mirror insertion:** one superposed delta tensor plus deterministic task context coordinates reconstructs multiple task-specific deltas on demand.

- PA175: DARE sparsification is a mandatory comparator.
- PA16: Parameter Superposition is the closest native control.
- Controls: independent dense, independent DARE, random-context PS with and without DARE, Hadamard contexts, and byte-identical native PS.

## Protocol

Task counts 2/4/8/16/32, D=1024, DARE drop .9, private fraction 0/.1, three audit seeds. Complete payload includes shared base, bound delta, context seed/IDs and metadata. See [PROTOCOL.json](PROTOCOL.json) for frozen gates and controls.

## Results

Pending.

## H/T/D/C/U

- **H:** DARE followed by PS unbinding increases task multiplicity per stored delta tensor without unacceptable interference.
- **T:** Synthetic linear task deltas, held-out probes, independent DARE and standard PS controls.
- **D:** Pending.
- **C:** Native Parameter Superposition may fully explain the View; context interference may erase the apparent storage gain.
- **U:** Pretrained fine-tuning, natural task performance, and optimized hardware runtime.
