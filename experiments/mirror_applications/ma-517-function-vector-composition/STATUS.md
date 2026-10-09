# MA-517 status

- Status: **FAIL**
- Branch: `research/ma-517-function-vector-composition-20261008`
- Protocol frozen before development: yes; Amendment 1 fixed the no-state serializer and Amendment 2 added composition-build timing. Earlier runs are preserved.
- Development seeds 51701/51702: complete; deterministic replay exact.
- Fresh seeds 51711–51713: sealed and unopened.
- Registry/status board: FAIL after reconciliation.

## H / T / D / C / U

- **H:** product Views should improve held-out identity composition by >=0.10 over both explicit sum and difference in both worlds, fit <=0.50x explicit FV bytes, and beat native PCA product by the frozen attribution margin.
- **T:** pinned Pythia-70m, six inverse relation pairs, 48 held-out queries per world, seeds 51701/51702, no optimizer updates; no-intervention, operand, explicit arithmetic, and native PCA-product controls. See PROTOCOL.json and RESULTS_CORE.csv.
- **D:** FAIL. Quality thresholds miss, no-intervention is already high, and native PCA exactly aliases all product ranks. Product payloads are 8,456–29,576 B versus 34,446 B explicit; fresh remains sealed.
- **C:** constrained identity ranking is mostly solved by the base model, and arithmetic on activation vectors may not realize sequential function composition.
- **U:** general compositional reasoning, other models/tasks, larger candidate spaces, and deployment latency are untested.

## Evidence classification

- **Facts:** 26 serialized payloads replay with exact byte hashes; metrics, splits, and operand vectors replay exactly; native aliases hold at every rank; no fresh seed directories exist. Five tests pass.
- **Interpretation:** storage compression is measurable for this aligned screen, but neither the frozen quality criterion nor Mirror-specific attribution is established.
- **Hypothesis:** future composition experiments need tasks where no-intervention fails and a native operator control that is distinct from the proposed View.

## Amendments

Amendment 1 fixed the no-state serializer failure. Amendment 2 recorded build timing omitted by the first completed run. The initial serializer failure and both pre-amendment run sets remain preserved.
