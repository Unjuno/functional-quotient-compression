# MA-865 results

## Fact

- One common MLP base was trained on four rotated Digits tasks, then each task model was fine-tuned from that base. Development seed 83 selected base LR=.03 and task LR=.1. Fresh seeds: 401, 419, 433.
- Independent experts averaged .9660–.9667 accuracy across tasks; task soup averaged .6597–.6833.
- SVD task-delta accuracy and actual bytes:
  - Rank 1: .6688–.7951 mean accuracy at 22,101 B.
  - Rank 2: .7625–.8375 at 31,765 B.
  - Rank 3: .8701–.8819 at 41,365 B.
  - Rank 4: .9660–.9667 at 51,093 B.
- Independent four-model bank: 43,827 B. Exact base plus four full task deltas: 50,777 B. The rank-3 bank saves 5.6% bytes but loses 8.5–9.6 pp mean task accuracy. Rank 4 restores expert quality but is 16.6% larger than the independent bank.
- Random intrinsic bases used the same bytes per rank but mean task accuracy was .8417–.8444 at rank1, .8403–.8451 at rank2/3, and .8410–.8451 at rank4. SVD improves rank3 over random by about 3.0–3.7 pp and rank4 by about 12.2–12.6 pp, but no SVD rank meets both the quality and 25% storage gates.
- Native PCA coefficients and the identity Mirror encoding had exactly equal logits. Training used 680 base updates / 43,120 examples and 1,360 task updates / 86,240 examples. Full four-task query proxy: 3,409,920 MACs. Per-run training wall time was about 0.68 sec.
- Full per-seed metrics and actual serialized payload bytes are in RESULTS_CORE.csv and fresh_results.json.

## Interpretation

This common-task tournament shows a real but narrow native-SVD compression/quality frontier: rank 3 modestly reduces bytes while losing about nine percentage points; rank 4 restores accuracy but exceeds independent storage. The random basis is a useful lower-quality control, and task soup is poor for these rotation-specialized models. These results do not establish a Mirror advantage because its coefficient code exactly aliases native SVD coordinates.

## H / T / D / C / U

- **H:** A shared task-coordinate basis can represent four related tasks more compactly than independent models, while beating random intrinsic coordinates; Mirror may improve that frontier beyond native PCA/task vectors.
- **T:** Common 64→32→10 MLP base; four rotation tasks; base 40 epochs and per-task fine-tune 20 epochs; dev seed 83 selected LR .03/.1; fresh seeds 401/419/433; independent bank, task vectors, soup, SVD ranks 1–4, random ranks 1–4. Actual payload bytes, updates/examples, query MAC proxies, and wall time measured.
- **D:** **FAIL** for the preregistered Mirror-specific frontier. Native SVD rank3 gives only 5.6% storage savings at an 8.5–9.6 pp accuracy loss; rank4 is larger than independent experts. Mirror outputs are exactly native outputs.
- **C:** Four task deltas span nearly four distinct directions; reducing rank loses task-specific information, while full rank pays for nearly the full task bank. The subspace is useful geometry but not a strong storage quotient at this task count.
- **U:** More tasks for basis amortization, larger pretrained models, quantized task codes, other data distributions, and actual runtime benefits remain untested.
