# MA-533 status

- Status: **FAIL** (development gate).
- Branch: `research/ma-533-transcoder-feature-experts-20261009`
- Base commit: `c07cba1d`.
- Protocol frozen: `81639629`; amendments 1–9 and final source committed through `3c7f06c7`.
- Result commit: `06d74b8b` (pushed).
- Development complete: yes; fresh/audit opened: no.
- Results committed: yes.

## H — Hypothesis

The released SmolLM2-135M layer-8 top-k=128 skip-transcoder replaces native MLP output with raw relative MSE <=0.10 and mean cosine >=0.95 on held-out natural text.

## T — What ran

Pinned model `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, transcoder `651f51421f2e1aa8fbd907e02ef421d3da55ff6d`, layer 8; Wikitext-2 train split, immutable parquet revision `d575192455c5e98b8daed777574046264579cb09`; 32 unique aligned 128-token blocks (16 fit, 16 disjoint eval), 4,096 tokens total, CPU, zero optimizer updates. Controls: native MLP, skip-only, rank-128 affine, raw transcoder, fit-only global RMS scaling and fit-only linear amplitude head. Fresh seeds 53311/53312 on the separate validation split were not accessed.

## D — Decision

**FAIL.** Raw transcoder: MSE 0.80601/cosine 0.73807; fit-only amplitude head: raw MSE 0.14862/cosine 0.95616. Both miss the raw-MSE <=0.10 gate. Direction-only passes only when using the target output norm and is explicitly non-deployable. Fresh remains sealed.

## C — Strongest counter-hypothesis

The checkpoint may encode MLP output direction but not its tokenwise magnitude. The model card says input/output were normalized but does not define the convention; global-RMS scaling fails and a small fit-only magnitude head still misses raw quality. Separately, the rank-128 affine control nearly matches raw quality at 592,381 bytes vs 341,363,524 bytes for the transcoder.

## U — Still unknown

Other layers/checkpoints, exact training normalization, validation/fresh seed behavior, downstream task functions, and logical Mirror role multiplicity remain untested. This does not establish that all transcoders fail. MA-534 is a separate task-level hypothesis.

## Fact / interpretation / hypothesis

- **Fact:** corrected eval token hash `db806506ce595a04300fe2b5c1e0748eee5942928ba5f13d73b26eafbb2d86f8`; exact payload bytes/hashes and stripped-base reconstruction are in `runs/dev/seed_53301/PAYLOAD_MANIFEST.json`.
- **Fact:** standalone base without replaced MLP is 267,128,681 B; raw transcoder total is 608,492,205 B; rank-128 total is 267,721,062 B.
- **Fact:** transcoder compute is 87,803,559,936 MACs/eval vs native MLP 5,435,817,984 and rank-128 301,989,888.
- **Interpretation:** a small direction representation exists, but restoring useful output magnitude costs extra learned state and remains below the quality gate; the low-rank control provides a much cheaper near-quality approximation.
- **Hypothesis:** sparse feature views might still support specialized task roles when task-level output scales are allowed; MA-534 will test that with dense MLP and simple sparse-gating controls.

## Next action

Record MA-533 FAIL, leave validation sealed, and continue with MA-534 under a separately frozen task-quality protocol.
