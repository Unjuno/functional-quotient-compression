# MA-446 — Learned optimizer for Mirror coordinates only

## H — Hypothesis

A learned per-step diagonal update schedule on only a 2D functional coordinate improves held-out few-shot quality by at least 10% over tuned Adam and SGD at four updates, without larger serialized inference state.

## T — Test

2D latent linear regression, four support and 128 query samples. Development worlds 44600–44601 selected SGD lr 0.3 and Adam lr 0.1 and trained an 8-step coordinatewise schedule. Fresh worlds 44610–44612, seeds 0–2, 20 tasks/world/seed, evaluated 1/2/4/8 steps.

## D — FAIL

Step-4 mean NRMSE was 0.4062 learned, 0.4882 Adam, 0.5697 SGD. Learned beats Adam on average, but misses the per-world 0.90x criterion in world 44612 (0.5042 vs 0.5122). Exact serialized N=20 package was 101.25B/task learned vs 91.65B/task Adam.

## C — Strongest counter-hypothesis

The learned schedule overfits the development distribution; the slight third-world gain does not justify increased policy bytes.

## U — Unknown

No recurrent learned optimizer, nonlinear task, or natural-data evidence.

See protocol, fresh run artifacts, payload files and verification record.
