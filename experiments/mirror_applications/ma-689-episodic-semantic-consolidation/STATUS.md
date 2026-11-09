# MA-689 status

- Status: FAIL (Mirror-specific attribution; aligned synthetic storage point only)
- Branch: `research/ma-689-episodic-semantic-consolidation-20261109`
- Protocol frozen at `7715d2c7`; schema-only accounting amendment made before corrected development rerun
- Development: complete; seeds 68901/68902, rho 0/.05/.2
- Fresh/audit: sealed; exact native PCA alias observed
- Serialized replay tests: passed (2 tests)
- Results committed: `dee21d8aed8740cb1c5a5bf09b3eb82f8c5b4d6f`
- Registry row updated: pending integration

## H — Hypothesis

Transfer rank-4 episode coordinates into a persistent base and delete the episode store without losing task recall, using materially fewer complete bytes than independent episode adapters.

## T — Treatment and controls

32 synthetic episode vectors in R^32; 24 basis-fit episodes, 8 excluded episodes; two development seeds; rho=0/.05/.2. Independent full vectors, shared code bank, dense consolidated base, and native PCA code bank/base were serialized and replayed. Measured pre-transfer state, duplicate transfer peak and post-deletion base-only state, plus per-vector MAC proxy and CPU lookup time.

## D — Decision

FAIL for Mirror-specific attribution. rho=0 passed recall and storage with <7.5e-8 NRMSE at 1,797B versus a 3,782–3,798B independent bank. rho=.05/.2 missed the .02 recall gate at .0432–.0448 and .1728–.1790. Mirror and native PCA exactly match function, MACs and canonical serialized bytes. Fresh remained sealed under the frozen rule. Dense consolidation kept exact recall at 3,557–3,581B. Code lookup costs 128 MAC/vector versus direct full-vector lookup's zero; transfer peak was about 7.3KB.

## C — Strongest counter-hypothesis

Ordinary PCA plus moving its coefficients into a checkpoint explains the storage point. The coordinates remain stored per episode, so this run did not test semantic weight consolidation.

## U — Unconfirmed

No natural episodes, learned semantic integration, language model, online training, forget policy or hardware serving test.

## Next action

Integrate the FAIL result, registry/status/claim records and provenance into the reconciliation branch; then select the next untested P0 from the live status board.
