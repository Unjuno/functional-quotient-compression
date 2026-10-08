# MA-511 — Hierarchical condition x behavior codes

Status: SCREENING; dev runs not yet accessed.  
Branch: research/ma-511-hierarchical-condition-behavior-codes-20261008  
Prior art: PA101, Conditional Activation Steering.

## H — Hypothesis

Fit 48 of 64 synthetic condition-behavior function vectors, withholding 16 whole pairs while preserving all condition and behavior IDs in the fit graph. A rank-4 condition code plus rank-4 behavior code should compose the held-out outputs at relative RMSE <=.05 and use <=.50x the full explicit pair table bytes. To call this Mirror-specific, it must also beat native additive factorization by >=10% at equal quality.

Mirror insertion: independently encode condition and behavior factors in shared bases, then compose their decoded vectors for a requested pair. The pair query provides the two factor IDs to every method.

## T — Frozen protocol

See PROTOCOL.json and freeze.json. Two development seeds (51101, 51102), 64-dimensional outputs, 8 condition IDs, 8 behavior IDs, 48 visible pairs, and 16 held-out pair identities. Basis ranks {2,4,8}, with rho {0,.1,.25} private pair residual. The native additive least-squares plus PCA control receives the same visible vectors. Full-pair table, bases, factor codes, global bias and schema are charged in actual uncompressed NPZ bytes. Fresh seeds 51111–51113 stay sealed.

## D — Decision

Pending frozen development run.

## C — Strongest counter-hypothesis

A standard additive factor model trained on visible condition-behavior vectors already predicts unseen pairs. Its parameters and decoded outputs are algebraically the same as the proposed Mirror factorization, so a held-out composition result alone is not Mirror-specific.

## U — Scope limits

This is a synthetic activation-function composition mechanism test, not a frozen language model or semantic conditional-steering experiment. Pair counts are not independent capacity; only held-out output quality and serialized state count.

## Evidence classification

- Facts: pending frozen dev measurements.
- Interpretation: a successful held-out result would show additive compositional structure in this synthetic world; native factorization is a direct attribution control.
- Hypothesis: semantic LM behaviors may or may not factor condition from behavior; this screen does not answer that.
