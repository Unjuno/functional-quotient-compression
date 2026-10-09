# MA-546 — Functional invariance of hidden activation View symmetries

Status: SCREENING  
Branch: `research/ma-546-representation-symmetry-20261009`  
Base commit: `ac1c3e1814aada23eb99254b841faa31b4bb0ffb`  
Prior art: PA47 (monomial weight symmetries), PA97 (representation engineering)

## H — Hypothesis

An invertible reparameterization of the post-GELU layer-3 MLP activation, paired with its exact inverse in the down-projection, preserves Pythia outputs and adds no logical functions. If the compensation is omitted, output differences represent a changed function, not a symmetry.

## Insertion

At layer 3, transform the 2048D activation after GELU by a monomial or dense orthogonal matrix. Apply the inverse coordinate change to the matching down-projection. Compare output logits, top-1 agreement, divergence, code bytes and compute against the canonical model.

## T — Execution

Pending. Development seeds 54601/54602; fresh seeds 54611–54613 stay locked until the frozen implementation is committed. No parameter learning or hyperparameter selection.

## D — Decision

Pending.

## C — Strongest counter-hypothesis

A transformed coordinate system may be exactly output-equivalent only because the next linear operator receives the inverse transform. The address then names a gauge choice, not an additional behavior. Some uncompensated transforms may alter outputs without producing useful task behavior.

## U — Not established

Pending. The screen covers one MLP interface in one Pythia checkpoint and does not test attention, residual-stream layer norms or learned task-conditioned views.
