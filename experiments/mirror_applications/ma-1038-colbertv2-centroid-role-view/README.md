# MA-1038 — ColBERTv2 shared centroid bank with query-role views

Status: **FAIL**  
Evidence lane: **NATURAL_BENCHMARK_SCOPED / STORAGE / COMPUTE**  
Branch: `research/ma-1038-colbertv2-centroid-role-view-20261115`  
Protocol: `PROTOCOL.json` (frozen before NFCorpus audit metrics)  
Results: `RESULTS.md`; exact metrics and payload sizes: `runs/summary.json`  
Verification: `VERIFICATION.json`  
Provenance: `ARTIFACT_PROVENANCE.json`

## H — Hypothesis

Two query-length-role Givens views over one fixed residual-coded ColBERTv2 document bank will improve fresh-dataset nDCG@10 by at least 0.5 absolute percentage points over the same bank without views and beat a matched native query-side Givens control by at least 0.5 pp.

## T — What ran

- Official ColBERTv2 MS MARCO checkpoint, pinned to Hugging Face revision `c1e84128e85ef755c096a95bdb06b47793b13acf`; model archive SHA256 is in the provenance manifest.
- Frozen 128-D token encoder and exact MaxSim. One 256-centroid shared bank was fit on 100,000 sampled SciFact document-token vectors; per-coordinate residuals were encoded as signed int8 with train-derived scales.
- Development: 809 SciFact train-qrel queries over all 5,183 SciFact documents. Nine angles from −0.20 to +0.20 radians were evaluated separately for short and long query roles. The median role threshold was 22 active tokens; both roles selected +0.20 radians.
- Fresh audit: all 323 BEIR NFCorpus test-qrel queries over all 3,633 NFCorpus documents. No audit data selected the angles or codebook.
- Controls: full-float document index; shared residual-coded index without a view; the same two view angles applied as the native inverse Givens to query vectors. Independent query encoders were omitted because a full BERT encoder per role would add hundreds of MB and would not isolate the registered small-coordinate contribution.
- There were zero optimizer updates. The development sweep evaluated 7,281 role-angle/query cases; the audit evaluated 1,173,459 query-document pairs per arm.

## D — Decision

**FAIL for Mirror-specific utility and role multiplicity.** The Mirror setting improves audit nDCG@10 by only **0.076 pp** over residual coding without a view, below the frozen +0.5 pp gate. MRR@10 decreases by 0.108 pp. The two role angles are equal, so the training result did not produce two distinct learned functions; on NFCorpus only 1/323 audit queries falls in the long role. A native query-side inverse Givens reproduces the Mirror exactly (sampled token-logit maximum absolute error `7.15e-7`) and has the same quality and serialized bytes.

The residual code itself is effective in this screen: document-vector storage falls from 287,293,568 B of float32 vectors to 72,516,576 B for centroid IDs, int8 residuals, centroids and scales. Including ColBERTv2, tokenizer, document IDs and metadata, the complete full-float payload is 726,467,488 B; the complete residual-coded payload is 511,690,535 B. Adding the View and its role metadata raises this to 511,690,756 B (+221 B over the compressed native control). The 29.56% complete-payload reduction is therefore from ordinary residual compression, not Mirror.

## C — Strongest counter-hypothesis

The small nDCG increase is ordinary query-space Givens reparameterization and sampling variation, not a Mirror-specific benefit. This is directly supported by the exact native query-side control. The audit's 322:1 role imbalance also prevents a strong test of two independently useful query roles.

## U — Still unknown

- This is one pretrained checkpoint and one cross-domain audit. It does not establish performance on MS MARCO, PLAID, other corpora, GPU indexing, or production latency.
- The CPU full-MaxSim implementation is not PLAID centroid-pruned traversal.
- Per-arm search wall times varied substantially across repeated sequential replays; latency/QPS is **NOT ESTABLISHED**.
- Independent per-role ColBERT encoders and trained residual VQ alternatives were not run.
- The direct document-side rotation adds about 69.6 GFLOPs per full NFCorpus audit pass; it was checked against the algebraically equivalent query-side push-through on 4,096 fixed query/document token pairs rather than timed as a full second retrieval implementation.

## Results

See `RESULTS.md` for facts, interpretation, amendments and scope boundaries. Large model, token-vector and payload arrays remain outside Git; their byte sizes and hashes are recorded in `ARTIFACT_PROVENANCE.json`.
