# MA-429 — shared recurrent block with continuous depth Mirror views

## H — falsifiable hypothesis

A shared recurrent transition conditioned by a continuous Givens depth coordinate will extrapolate to unseen recurrence depths better per byte than a simple linear timestep embedding or discrete step-specific matrices at the same recurrence count.

## T — protocol

Fit a synthetic 2D recurrent trajectory whose teacher rotates one stable transition with depth. Train only on depths 1–4 and evaluate per-step state error through depth 8. Compare static shared, continuous Mirror timestep view, linear matrix time embedding, and four discrete step matrices clamped after depth 4. All methods run the same number of recurrence steps.

This is a recurrent dynamics proxy, not a Transformer language-model result.
