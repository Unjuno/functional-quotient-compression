# Rejected development-only XOR pilot

Before PA231 protocol review, an XOR toy used an alpha-interpolated two-endpoint tanh MLP (500 updates, lr=.05) on one synthetic seed. At beta=1, five alpha views had 97.1–97.6% accuracy and mean pairwise disagreement .00996; beta=0 reached 97.3–97.6% with disagreement .00518; five independent models averaged 97.55% accuracy and .00820 disagreement. Actual endpoint/code payload was 3,977 B versus 7,719 B independent models.

No fresh seed was accessed. This prototype did not follow PA231 data/model protocol sufficiently and is excluded from MA-856 evidence. It is preserved only as an exploratory scoping note; the registered experiment uses sklearn Digits and PA231 Algorithm 1.
