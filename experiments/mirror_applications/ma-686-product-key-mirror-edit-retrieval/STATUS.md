# MA-686 status

- Status: FAIL (quality/scope gates and Mirror-specific claim)
- Branch: `research/ma-686-product-key-mirror-edit-retrieval-20261104`
- Protocol frozen before development: yes
- Development: seeds 68601/68602; noise 0/.25
- Fresh/audit: sealed

## H — Hypothesis

Product-key retrieval over factorized subject/relation keys can retrieve edit codes with lower search work than flat scanning, while shared-basis Mirror values reduce payload without harming edit quality or scope safety. Mirror must beat native product-key low-rank values.

## T — Treatment and controls

Synthetic 32×32 product-key address space with 256 active edits, 4 scope domains, key dimension 8, delta dimension 32, rank-4 shared delta basis; two seeds. Compared flat full-key scan/full values, product key with direct values, product key with shared basis+codes, and native low-rank product-key control. Queries included active edits, same-key wrong-scope edits, and absent pairs; Gaussian key noise 0 and .25. Metrics are replayed from serialized NPZ payloads.

## D — Decision

FAIL. At noise 0, all methods retrieved active edits exactly, but the flat bank falsely activated 21.9–26.2% of absent queries; product-key variants had zero absent false activations and zero wrong-scope activations. At noise .25, flat recall was .875/.891, product-key recall .652/.711; product-key wrong-scope activation was .016–.023 and false activation .016–.023, above the 1% gates. Mirror product-key payloads were ~8.5KB versus ~34.3KB flat; compute proxy ~491k versus 3.15M. Native low-rank product-key exactly matched Mirror in outputs, retrieval, bytes and compute. Fresh stayed sealed.

## C — Strongest counter-hypothesis

Product-key factorization itself explains the storage/search tradeoff. At moderate key noise, independent subkey retrieval reduces recall; native low-rank values reproduce every Mirror result.

## U — Unconfirmed

No trained edit system, natural query embeddings, learned threshold tuning, lifelong updates, or hardware latency is measured. Synthetic key noise is not calibrated to real retrieval distributions.

## Evidence labels

- Fact: serialized payload replay results, edit metrics, false-scope/absent activation, bytes and MAC proxies are in `RESULTS_CORE.csv` and `source/development.json`.
- Interpretation: factorized product keys cut storage and scan work in this synthetic case but suffer a retrieval-quality/scope safety tradeoff under key noise.
- Hypothesis: learned subkey geometry or additional scope-aware reranking may recover quality, but any design must beat native Product Key Memory controls.
