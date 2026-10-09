# MA-534 status

- Status: **FAIL** (development quality gate).
- Branch: `research/ma-534-transcoder-logical-mlp-20261009`
- Protocol frozen: `930a1325`; tests/serialization fix: `bcf2e106`; timing amendment: `b59c1cd2`.
- Development complete: yes; fresh/audit opened: no.
- Results committed: pending final result commit.

## H — Hypothesis

Four input-cluster roles using one shared 32-atom transcoder bank plus 16 per-role Givens angles can reduce held-out raw layer-8 MLP-output error by >=10% versus plain role sparse gates and diagonal gains, while paying only a small role code.

## T — What ran

Pinned SmolLM2-135M `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` and skip-transcoder `651f51421f2e1aa8fbd907e02ef421d3da55ff6d`; Wikitext-2 train split revision `d575192455c5e98b8daed777574046264579cb09`; seed 53401, 32 unique aligned 128-token blocks (16 fit,16 evaluation; 4,096 tokens), four fit-only nearest-centroid roles, 32 shared bank features, top-8 decode. Givens angles and diagonal gains each got 200 Adam updates; plain supports were fit-only. Controls: native MLP, full top-128 transcoder, hard-tied shared bank, role-specific sparse gate, diagonal gain, Givens Mirror, four copied full MLPs. Validation seeds 53411/53412 were not run.

## D — Decision

**FAIL.** Mirror MSE 0.955803 vs plain gate 0.959720 (only 0.41% lower, below the frozen 10% margin) and diagonal control 0.940216 (Mirror is 1.66% worse). Worst-role/mean Mirror MSE ratio is 1.019, within the 1.25 limit; the failure is the required cheap-control margin. Fresh validation stays sealed.

## C — Strongest counter-hypothesis

The 32-atom bank drops important dense-MLP directions. The full top-128 transcoder MSE is 0.796, while every top-8 bank method is 0.94–0.96. Learning diagonal gains recovers more than Givens views, so role-conditioned output variation may be mostly coefficient scaling and not an orthogonal-coordinate effect.

## U — Still unknown

Semantic meaning of the four input clusters, validation/fresh-domain behavior, downstream next-token loss, larger banks, and other model/layer/checkpoint combinations remain untested. This is not a general claim about all transcoders or sparse circuits.

## Fact / interpretation / hypothesis

- **Fact:** token hash `629188f983ac74f22bc62505afe45c183950b765f7403a3f89102b6291d97acb`; role counts `[890, 216, 214, 728]`. All metrics are in `runs/dev/seed_53401/summary.json`.
- **Fact:** actual incremental/standalone Mirror payload is 1,487,262/268,615,943 B; full model 272,437,465 B. Mirror angles were nonzero (mean absolute 0.509 rad).
- **Fact:** selected-bank compute is 354,816 MAC/token plus 2,304 router MACs for role modes; native MLP is 2,654,208. Inference wall: Mirror 0.0348s, sparse gate 0.0320s, diagonal 0.0333s, native 0.1231s per 2,048 vectors.
- **Interpretation:** storage and compute improve modestly with an unacceptable output-quality loss; Mirror did not improve the fixed-byte quality frontier over diagonal gain.
- **Hypothesis:** role-specific feature codes need either a richer/private atom basis or target-function-aware atom selection; the same 32 atoms do not give enough useful role diversity.

## Next action

Record FAIL and pause the same pretrained-transcoder feature family after two related failures; continue with the next distinct P0 candidate MA-539.
