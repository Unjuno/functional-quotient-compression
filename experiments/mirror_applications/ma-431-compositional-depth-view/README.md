# MA-431 — compositional Mirror views over recurrent depth

Status: **PROMISING, scoped synthetic mechanism/storage-quality screen; compute cost increased.**
Evidence lane: MECHANISM / STORAGE / RUNTIME.
Base commit: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`.
Prior art: PA72, Universal Transformer.

## H / T / D / C / U

**H:** For sequences of noncommuting depth operations, a shared recurrent block with a compact Mirror coordinate at each step can recover held-out ordered compositions with fewer serialized bytes than independent step matrices, and improve the quality/byte frontier over simple low-rank gates.

**T:** CPU float32 PyTorch. A shared 16×16 recurrent transition; two noncommuting 2D shear operations embedded in a 16D state; train on lengths 1–6 while holding out a fixed set of length-6 ordered programs. Compare static tying, rank-1 gate, Universal-Transformer-style depth-conditioned rank-2 gate, Givens-conjugated Mirror, and independent per-symbol matrices. Development: worlds 43100–43101 × seeds 0–2, LR `{0.001, 0.003}`, Mirror scale `{0.5, 1.0}`, 1,500 updates. Fresh: worlds 43110–43112 × seeds 0–2, selected LR 0.001 and Mirror scale 1.0, 1,500 updates. Fresh settings were locked in `DEV_SELECTION.json` before the run.

**D:** **PROMISING, scoped to this teacher family.** In every fresh world, Mirror length-6 relative MSE was at most 1.05× independent full per-symbol matrices and payload was 2,921 B versus 3,945 B (25.9% fewer bytes). Across all 9 fresh runs, mean relative MSE was 4.50e-6 for Mirror versus 7.09e-6 independent. Mirror also dominated the rank-1 gate on quality, payload and MAC proxy. It did not improve compute: Mirror used 2,304 MAC proxy/example versus 1,536 for independent; mean training time was 5.01 s versus 3.03 s. Single-batch CPU throughput was variable and is only descriptive. Universal low-rank obtained similar mean error (4.39e-6) at 3,993 B and 8,064 MAC proxy/example, leaving a higher-quality/compute-cost Pareto point.

**C:** The teacher was deliberately generated as two exact Givens-conjugate shears. This is a best-case aligned orbit for the Mirror coordinate; the result may not transfer to naturally arising recurrent operators. The Universal low-rank control is slightly more accurate, and the current CPU timing is noisy.

**U:** Natural data and language tasks, larger state and block sizes, learned operation routing, codes not supplied at inference, non-conjugate operators, optimized fused kernels, and near-convergence/fixed-byte capacity are untested.

## Mirror insertion

> **Mirror insertion:** this experiment adds a scalar angle coordinate `m_t` to the shared recurrent transition through a Givens-conjugated view `R(m_t) W R(-m_t)`, so distinct ordered logical step operators can be applied without storing a separate full matrix at every depth.

- Native method: Universal-Transformer-style repeated shared transition with timestep conditioning.
- Persistent `m_t`: deterministic two-symbol angle address (0 or π/2); codebook and metadata are included in each serialized payload.
- Cheapest controls: a rank-1 residual/gate and a Universal-Transformer-style shared rank-2 depth/symbol gate.
- Symbol IDs are supplied equally to every method; this experiment does not measure a router.

## Facts

- Fresh rows: 45 (3 worlds × 3 seeds × 5 methods); each run used 1,500 updates and 192,000 training examples.
- Mean fresh payloads: Mirror 2,921 B; independent 3,945 B; rank-1 3,425 B; Universal low-rank 3,993 B.
- Mean fresh relative MSE: Mirror 4.50e-6; independent 7.09e-6; rank-1 9.73e-4; Universal low-rank 4.39e-6.
- Mean active MAC proxy/example: Mirror 2,304; independent 1,536; rank-1 3,264; Universal low-rank 8,064.
- Serialized payload size and SHA-256 were computed from `torch.save` bytes for every fresh model. All payload components include state tensors, angle/symbol metadata and inference configuration.

## Interpretation

The structured view recovered this intentionally conjugate operator pair with fewer bytes than independent matrices and lower error than the rank-1 gate. The storage/quality point costs more recurrent-step computation and training time than the independent reference. The rank-2 Universal Transformer control reaches similar quality with larger state and MACs, so the evidence supports a narrow structured-view Pareto improvement, not general recurrent-depth multiplicity.

## Hypothesis boundary

The result does not establish natural sequence utility, general capacity, independent function count, faster inference, or an advantage on arbitrary operator families. Any extension must preserve full serialized bytes, measured compute, held-out program identities, and direct timestep/low-rank controls.
