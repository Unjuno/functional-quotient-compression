# MA-517 — Mirror composition of function vectors

Status: FAIL (pre-registered screening gates not met).

## H / T / D / C / U

**H:** Two separately extracted task vectors for fruit→color and color→shape may compose better with role-specific layer placement than raw same-layer vector addition.

**T:** Pinned frozen GPT-2; synthetic mapping tasks with four support pairs for each function and four held-out fruit→shape queries. Compare query-only, direct combined ICL, raw `f1+f2` at layer 6, layer-factorized `f1@4/f2@8`, and each vector ablation. Fresh worlds 51710-51712 × seeds 0-2. Both vectors and all metadata are charged.

**D:** FAIL. Across all three fresh worlds, direct composition ICL, query-only, raw sum, factorized layers, and both single-vector ablations all scored 0% on exact next-token accuracy. Direct-ICL target NLL ranged 6.27–6.90; factorized target NLL ranged 9.78–10.00. The two extracted task vectors were nearly indistinguishable across tasks (mean cosine 0.99968). The two-vector serialized payload was 99,945 bytes for raw and factorized arrangements (50,793 bytes for either single-vector ablation); pinned model runtime files were 550,959,861 bytes. Extraction averaged about 9.2 seconds per 16-task bank, and intervention evaluation averaged about 1.2–1.4 seconds per method/bank on CPU.

**C:** The direct in-context control also failed at 0%, so this screen did not establish whether composition itself fails. A strong candidate cause is invalid task serialization/format sensitivity for GPT-2 on the synthetic two-hop prompt; the near-identical deltas also indicate the chosen neutral-vs-demo contrast did not isolate task identity.

**U:** Whether this composition works with a validated native ICL format, whether task vectors can be extracted with a causal procedure, behavior on natural tasks or larger LMs, and learned structured Mirror codes. No quality/capacity/storage frontier claim is supported.
