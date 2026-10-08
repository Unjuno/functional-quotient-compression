# MA-962 — Task × time-step Mirror normalization View

Status: **SCREENING — protocol frozen before data acquisition**  
Branch: `research/ma-962-task-time-mirror-tebn-20261008`  
Baseline: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`  
Draw 15 selected MA-962 uniformly from 553 eligible P0/UNTESTED candidates. The exact ordered pool, exclusions, seed, index and SHA-256 are in `source/`.

## H — Falsifiable hypothesis

For four task-specific temporal encodings of MNIST, one shared SNN with a learned temporal gain profile and one scalar cyclic phase code per task will stay within 1.5 percentage points of native task×time TEBN accuracy at at least 2% fewer actual serialized inference bytes (<=0.98x), and beat a byte-near rank-1 task×time gain factorization by at least 1 percentage point.

## Prior art and Mirror delta

PA283 TEBN already learns time-step-specific normalization scaling. The comparison therefore charges the complete task×time gain table as the native control. The Mirror candidate uses a shared time×hidden gain profile and one continuous phase coordinate per task; periodic linear interpolation shifts the profile in time. It does not change the shared synaptic weights. The ordinary rank-1 task×time factorization is included to determine whether any improvement is specific to the phase View.

The twelfth-sweep notes also warn that task labels must be treated consistently, spike count and timing precision matter, and temporal-gain compression is not novel by itself.

## Frozen protocol

One hidden-layer LIF network (784 input, 128 hidden, 10 outputs), 8 time steps, 4 cyclically shifted rate-coding tasks. MNIST official training examples are split into 50,000 train and 10,000 development rows; the official test files are locked until the development gates pass. Conditions: shared no-task gain, native task×time TEBN, cyclic-phase Mirror, rank-1 task×time gain control, and independent SNN per task. Two development seeds; 800 updates per shared condition and 800 updates per independent task. Full parameters and metadata are charged by actual serialized payload bytes.

Shape-only accounting before data shows the duplicated task×time gain table is a small share of the total serialized model; even perfect gain-code compression saves only about 3% end-to-end. The protocol therefore requires a smaller but real >=2% complete-payload reduction. Exact hyperparameters, input envelopes, task phase offsets, seeds, amended gate thresholds, and audit rules are in `PROTOCOL.json`. Protocol and implementation will be committed before downloading or decoding MNIST.

## Decision and limits

A fixed-budget pass is a mechanism/learning-efficiency signal only. It is not a capacity or neuromorphic hardware claim. Any failure against TEBN or the rank-1 control will be recorded as such, with the audit left unopened unless the frozen development gates pass.
