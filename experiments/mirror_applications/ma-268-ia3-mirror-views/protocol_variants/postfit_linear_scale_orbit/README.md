# MA-268 — IA3 activation scaling vs Mirror activation views

Status: SCREENING. Evidence lane: MECHANISM / STORAGE. Base: `f7f76de193063950d28b7834f337840b1e86f0ce`.

## H / Mirror insertion

H: For task functions formed by an orthogonal orbit of hidden-channel gains, one shared activation scale plus per-task pairwise rotation coordinates can retain quality with fewer bytes than IA3 per-channel scales; arbitrary gains should require more private coordinates.

> **Mirror insertion:** this experiment adds task-specific pairwise rotation angles `m_t` to a shared hidden activation gain vector so multiple logical activation functions reuse one physical shared matrix and base gain.

PA19 says IA3 per-channel activation scales are the minimum-cost control. Compare shared weights with hard-tied gains, IA3 per-task gain vectors, Mirror rotation views, and independent task matrices.

## T

Synthetic regression through a frozen 12-hidden-channel by 6-output linear map, eight tasks. Aligned scales rotate a common 12-vector within six channel pairs; stress scales are independent random vectors. 512 train and 1024 test examples/task. Fit post-hoc gains from train examples. Development seeds 21/39; fresh seeds 111/233/317/431 are frozen in protocol. Zero optimizer updates.

## Gates

PASS: at least 3/4 fresh aligned worlds have Mirror MSE <=1.10x IA3, payload <=80% IA3, and lower MSE than shared. FAIL: Mirror exceeds IA3 MSE by 25% in >=3/4 aligned worlds, or matched-quality byte saving is <10%.

Actual uncompressed NPZ bytes include shared W, IA3 vectors, Mirror base gains and packed angles, and independent matrices. Report examples, optimizer updates, active MAC proxy and evaluation timing; matrix materialization is not runtime-optimized.

## H / T / D / C / U

FACT: pending.
INTERPRETATION: pending.
HYPOTHESIS: pending.
COUNTER-HYPOTHESIS: pending.
UNCONFIRMED: pending.
BOUNDARY: synthetic frozen linear block only; no LM, training or capacity claim.

## Results — H / T / D / C / U

FACT: Across fresh seeds 111, 233, 317, 431, aligned-orbit mean MSE was 2.70e-24 for Mirror and 2.12e-24 for IA3, with payloads of 1,550 and 1,838 bytes respectively. This is a 15.7% byte reduction, below the predeclared 20% PASS gate. Shared gains had MSE 0.876 at 1,166 bytes; independent full matrices had numerical-zero MSE at 4,866 bytes. On independent random gains, IA3 MSE was 2.78e-24 and Mirror MSE 0.210. Each seed/condition used 4,096 fit examples and zero optimizer updates. Serialized state roundtrip reconstructed all operators exactly.

INTERPRETATION: On a deliberately aligned activation orbit, the View matched IA3 with a measurable but sub-gate byte reduction. Arbitrary per-channel gains are not recovered; sharing gives a quality tradeoff.

HYPOTHESIS: Structured activation Views may be worthwhile when role changes follow a low-dimensional orthogonal orbit and task-code storage is a meaningful portion of the payload.

COUNTER-HYPOTHESIS: IA3 is already a small and direct code; basis/weight bytes limit the overall reduction, and a quantized IA3 vector could narrow the difference.

UNCONFIRMED: trained adaptation, real Transformer activations, task interference, activation intervention costs, quantized IA3, runtime, and fixed-byte task scaling.

Decision: PROMISING, scoped to aligned synthetic activation functions; formal 20% storage gate missed, so not a PASS or adoption claim.
