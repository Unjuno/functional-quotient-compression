# MA-712 — DARE before parameter-superposition unbinding

Status: **FAIL at development; audit unopened**  
Evidence lane: MECHANISM/STORAGE/QUALITY  
Base commit: `c3521fb2`

## Hypothesis

H: DARE sparsification followed by context binding into one shared delta tensor can retain per-task held-out function quality at lower complete serialized state than independent DARE deltas; structured Mirror contexts must beat standard Parameter Superposition contexts at matched bytes.

## Mirror insertion

> **Mirror insertion:** one superposed delta tensor plus deterministic per-task context coordinates reconstructs several task-specific deltas on demand.

PA175 makes DARE sparsification mandatory. PA16 is the closest native comparator. Controls are independent dense, independent DARE, random-context PS with and without DARE, Hadamard contexts, and byte-identical native PS. Complete payloads include the shared base, tensor state, context seed, task IDs and metadata.

## Results

**FACT:** Development seed 71201, D=1024, task counts 2/4/8/16/32, DARE keep fraction .1. At rho=0 and N=8, independent DARE payload was 10,156B with task-output NRMSE 2.808; DARE+random-PS was 9,749B with NRMSE 8.692; DARE+Hadamard was 9,749B with NRMSE 8.785. At N=32, independent DARE was 24,508B/2.919 NRMSE; DARE+random-PS and DARE+Hadamard were both 9,797B with NRMSE 17.599/18.050. At N=2, PS state (9,737B) exceeded independent DARE (6,514B). The no-DARE random PS baseline at N=8 was 9,749B/2.633 NRMSE; at N=32 it was 9,797B/5.524. Across task counts, Hadamard Mirror payload was byte/function-identical to native Hadamard PS. At private delta fraction .1, trends were similar.

**INTERPRETATION:** Binding compresses stored state once task count is large enough, but unbinding interference grows sharply with task count and destroys per-task function quality. DARE before binding did not prevent interference. The small apparent N=8 byte reduction is only ~4%, far below the frozen 25% gate. Hadamard did not improve over random PS and exactly aliases the native method.

**HYPOTHESIS:** Parameter superposition can reduce physical delta storage, but independent task behavior needs more capacity or a better unbinding code; the tested context views do not provide it.

**BOUNDARY:** Synthetic linear task deltas and random probes only. No pretrained model, real fine-tuning, language benchmark, or GPU runtime. Audit seeds remained unopened after the development quality/byte gates failed.

## H/T/D/C/U

- **H:** DARE followed by PS unbinding increases useful task multiplicity per stored delta tensor.
- **T:** Development seed 71201; D=1024, task counts 2–32, DARE 90% drop, private fraction 0/.1; independent dense/DARE, random and Hadamard PS, with and without DARE.
- **D:** **FAIL at development.** N>=8 saves state bytes but NRMSE rises severely; at N=2 bytes are larger. The <=75% byte gate is missed at N=8, and Hadamard exactly aliases native PS.
- **C:** Standard Parameter Superposition entirely explains the binding/unbinding function; context interference dominates storage benefit.
- **U:** Whether trained task deltas or learned context codes change the interference frontier. Audit unopened; no natural task/runtime claim.
