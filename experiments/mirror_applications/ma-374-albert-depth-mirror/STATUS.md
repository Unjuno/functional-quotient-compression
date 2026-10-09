# MA-374 status

- Last verified commit: `44a169d0`

- Status: **FAIL** (bounded digits residual-block proxy)
- Branch: `research/ma-374-albert-depth-mirror-20261009`
- Protocol frozen before development: yes; selected LR 0.003 on worlds 37400/37401
- Fresh worlds 37410/37411/37412 evaluated once at locked LR 0.003
- 42 result rows replayed; current-workspace payload hashes and exact serialization outputs checked
- 3 unit tests pass
- Next candidate: MA-375 (one-shot supernet + Mirror correction)
- Earlier duplicate-branch recurrence screen is archived as unregistered development-only evidence; it does not change this FAIL.

## H — hypothesis

A depth-specific four-angle Givens coordinate on a fully tied repeated residual block restores useful layer roles versus hard tying, stays near untied quality at <=60% bytes, and outperforms equal-code FiLM.

## T — execution

Three applications of a residual block with an attention-like linear map and FFN-like two-layer map; CPU digits, stratified 60/20/20 per world; 600 training updates and 100 view updates. Compared untied, fully tied, A-only shared, F-only shared, fully tied+Mirror, and fully tied+FiLM. Development worlds 37400/37401 selected LR 0.003; fresh 37410/37411/37412.

## D — decision

**FAIL.** Fresh mean accuracy: untied 97.22%, fully tied 96.48%, attention-shared 97.13%, FFN-shared 97.50%, Mirror 96.39%, FiLM 96.67%. Mirror does not improve over tied and is slightly worse than FiLM at equal payload (89,282 bytes). It is compact relative to untied (89,282 vs 257,447 bytes), but the gain is from hard block sharing and does not restore layer specialization. Partial sharing outperforms Mirror.

## C — strongest counter-hypothesis

Only sharing one of the two block components preserves useful depth-specific capacity; a small post-block rotation cannot recover what full attention+FFN tying removes.

## U — unresolved

This is not an ALBERT/BERT reproduction; no language modeling, long context, or hardware latency test. Fixed updates do not establish capacity.

## Evidence separation

- **Fact:** 42 rows replay, payload hashes match, metric replay max difference 4.7e-9, serialization logits exact.
- **Interpretation:** component-wise hard sharing is a stronger storage/quality tradeoff than adding the tested depth Mirror.
- **Hypothesis:** depth codes inserted between attention and FFN may behave differently; this requires a separate frozen MA, not a reinterpretation of this result.
