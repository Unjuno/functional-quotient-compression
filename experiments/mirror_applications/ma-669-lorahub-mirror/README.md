# MA-669 — LoRAHub coefficients from Mirror task code

## H — hypothesis

A development-fitted map from few-shot support statistics to adapter coefficients may replace gradient-free coefficient search with fewer evaluations, while preserving query quality. Compare LoRAHub and direct least squares.

## T — planned test

Frozen design in `PROTOCOL.json`. Draw44 selected MA-669 uniformly from the eligible P0 pool. PA148 describes LoRAHub's gradient-free coefficient search. The planned screen is synthetic and does not claim transfer to pretrained adapters.

## D — status

SCREENING; protocol frozen before fresh tasks.

## C — strongest counter-hypothesis

A direct least-squares coefficient solve is simpler and already optimal for this linear task; a Mirror code map may merely learn the same operation with extra state.

## U — unconfirmed

No implementation or measurements yet.

## Fact / Interpretation / Hypothesis

- Fact: LoRAHub searches scalar coefficients over existing adapters.
- Interpretation: Mirror must beat that search and compare against direct coefficient fitting.
- Hypothesis: a reusable support code may reduce adaptation compute on new tasks.
