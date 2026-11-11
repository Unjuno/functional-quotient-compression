# MA-824 — topology × node-function code

## H / T / D / C / U

**H:** Factorizing topology and node-function identity allows heldout architecture combinations to execute correctly with fewer complete bytes and active work than independent graph programs, beyond native modular routing.

**T:** Synthetic 8-node, 4-feature graph substrate; six topologies × four node functions; six heldout pairings; 10 edges/program; one message aggregation with tanh; seed 82401; zero updates. Compared independent pair networks, native modular routing, Mirror factorized views and fixed topology/function.

**D: FAIL.** Mirror and native modular routing both reach zero heldout error and execute 10 edge operations. Mirror uses 1,244 B; native uses 1,426 B. The independent pair payload is smaller at 1,042 B. Fixed topology/function is 751 B but misses quality (nRMSE .266). Thus the factorization composes, but it does not improve full bytes over the independent archive and exactly uses the ordinary stored topology/function IDs. The 182 B Mirror/native difference comes from duplicate metadata layout and is not a functional gain.

**Evaluation boundary:** heldout pair IDs and constituent topology/function descriptions are present in both inference payloads; this is not unseen topology or function inference.

**C:** Native modular routing implements the same composition; complete independent payload compresses well because of repeated synthetic matrices.

**U:** Learned graph programs, natural unseen combinations, dynamic topology inference and physical runtime. Fresh remained sealed.

## Evidence separation

- **Fact:** Sizes, hashes, heldout error, active edge operations and replay in `RESULTS_CORE.csv`.
- **Interpretation:** Recombining stored components works, but no storage gain over the independent payload or Mirror-specific mechanism is established.
- **Hypothesis:** More diverse learned node functions or larger graph sets may change the frontier; native routing remains mandatory.
