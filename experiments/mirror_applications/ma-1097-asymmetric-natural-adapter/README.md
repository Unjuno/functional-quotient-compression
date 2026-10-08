# MA-1097 — asymmetric natural adapter views

Status: SCREENING  
Evidence lane: STORAGE / MECHANISM  
Base commit: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`

## H — falsifiable hypothesis

Across independently fine-tuned rank-1 BERT adapters, a shared output-side (LoRA B) subspace will retain held-out task updates better than a byte-matched shared input-side (LoRA A) subspace, and the decoded bank will use at most 80% of independent LoRA payload bytes at no more than 10% relative update error.

## Selected candidate and draw

Draw24 sampled MA-1097 uniformly from the frozen eligible P0/UNTESTED pool. Pool size 532; SHA-256 `46e27c9dba61cd00113e3c385ecd1b60f7ca03dbad79fdda5a355a9906866abe`; seed `5349b37014f408ac170c08c2cc22fa201e824659adda893d0aba9d409d678e4c`; zero-based index 485. Live branch and existing-directory exclusions were recorded from the pre-run snapshot in `PROTOCOL.json`.

## Data and scope

Three public TransferGraph adapters for tweet_eval irony, emotion, and hate are used as natural learned updates. Their cards identify `bert-base-uncased`, LoRA rank 1, and the same 12-layer query/value modules. The cards do not pin the base model revision used during training; the current official base revision is pinned for reproduction, but exact training-time identity is unverified. This limits any cross-adapter interpretation.

The test is an oracle weight-space screen, not model fine-tuning or downstream task evaluation. Whole adapters are held out in three leave-one-task-out folds. Shared bases use only the other two adapters in each fold. No audit result is used for tuning.

## T — comparisons and measurements

For each of 24 matched query/value matrices, compare exact independent rank-1 factors, shared output B basis, shared input A basis, and two-sided shared U/V with dense per-task core. Test k=1 and 2; fit bases from training-task updates only. Compare held-out relative Frobenius update error, serialized bank bytes (all factors, heads, codes, bases and metadata), reconstruction equality, factorization/decode operations, and elapsed time.

## D — decision gates

- **PASS screening hypothesis:** output-side error is at least 10% lower than input-side median error, held-out relative error is ≤10% in every fold, actual serialized payload is ≤80% of independent LoRA, and it is not dominated by the dense-core control.
- **FAIL:** any numeric gate fails, or the output-side advantage disappears under gauge perturbation.
- **NOT ESTABLISHED:** data provenance or implementation invalidates a task comparison. No capacity or downstream-quality claim can be made by this screen.

## C — strongest counter-hypothesis

Any observed output/input asymmetry is task-bank-specific and is explained by ordinary shared-subspace projection; native dense-core compression or an independent LoRA remains the better quality/byte point.

## U — unresolved

Exact training-time base revision, downstream task scores after decoding, training a new task code from examples, natural task diversity beyond this three-task family, runtime on accelerator hardware, and marginal Mirror value over CtS/EigenLoRAx are not established here.

See `PROTOCOL.json`, `STATUS.md`, `RESULTS_CORE.csv`, and `VERIFICATION.json` for frozen details and measurements.
