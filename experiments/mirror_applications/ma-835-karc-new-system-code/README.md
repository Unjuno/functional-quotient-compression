# MA-835 — Few-shot code adaptation for a new KARC dynamical system

Status: SCREENING  
Evidence lane: MECHANISM / STORAGE / CONTINUAL DYNAMICS  
Base commit: `6e9572a2`  
Prior art: PA223, Kolmogorov-Arnold Reservoir Computing.

## Hypothesis

H: A compact system-specific Mirror coordinate over a frozen KARC feature bank will predict a newly encountered synthetic dynamical system with at least 25% fewer complete inference bytes than independent readouts, while preserving heldout rollout error and strictly beating native readout adaptation.

## Mirror insertion

- Native method: fixed KARC basis features with a closed-form system-specific linear readout.
- `m`: support-derived system code/readout coefficients selecting a function in the frozen basis.
- Persistent: one readout/code per encountered system.
- Cheapest ordinary equivalent: native closed-form readout adaptation.

## Physical-to-logical claim

- Shared physical object: frozen feature/basis bank.
- View: compact system-specific readout coordinate.
- Logical multiplicity: distinct system dynamics over one feature bank.
- Failure mechanism: readout coordinates may simply be the native KARC readout; shifted systems may require private basis growth.

## Prior-art delta

PA223 already establishes explicit basis functions plus a trained linear readout as a lightweight reservoir method. MA-835 tests whether a Mirror address provides a distinct storage/quality advantage for new systems before adding physical basis features. Native readout fitting is the required attribution control.

## Comparisons and gates

The preregistered controls are the shared-only bank, independent readouts, native closed-form readouts, Mirror-coded readouts, and private basis growth. PASS requires nRMSE <=0.02, >=25% fewer complete bytes than independent storage, and strict advantage over the native control. The exact FAIL conditions are in [PROTOCOL.json](PROTOCOL.json).

## Tuning boundary

One deterministic development seed, no optimizer updates, and no hyperparameter tuning. No fresh seeds are opened after an exact native alias or gate miss.

## Results

See [RESULTS_CORE.csv](RESULTS_CORE.csv), [STATUS.md](STATUS.md), and [VERIFICATION.json](VERIFICATION.json). Rank-2 Mirror and native factorization have identical 1,469 B payloads and nRMSE 0.03862, above the 0.02 gate. Independent readouts reach 0.000228 at 2,690 B; basis growth reaches 8.45e-9 at 3,206 B. Synthetic four-dimensional dynamics only; system readouts are stored and no unseen-system generalization is tested.
