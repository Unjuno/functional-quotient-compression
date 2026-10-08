# MA-990 — Shared HRTF field plus sparse Mirror listener code

Status: FAIL for Mirror-specific advantage and registered retrieval quality gate
Evidence lane: MECHANISM / STORAGE / RUNTIME / REAL DATA
Base commit: `f255f0b` (`research/mirror-application-worker-ready-20261007`)

## H — hypothesis

A shared data-derived HRTF field plus a small listener-specific sparse code can reconstruct held-out listeners from 3 or 5 measured directions with quality near retrieval, while reducing total serialized bytes once the shared field is amortized across enough listeners. Ordinary sparse PCA coefficients may match Mirror exactly.

## Mirror insertion

> **Mirror insertion:** add a sparse listener coordinate `m` to a shared mean HRTF field and rank-8 cross-listener residual basis, so one physical field can produce listener-specific full-sphere HRTF maps without storing one full map per listener.

The shared field is fit only on 30 training listeners. The four active basis atoms are selected using only sparse calibration directions. The matched ordinary top-4 sparse PCA control uses exactly the same fitted code. All shared basis, mean, code values, indices, metadata, retrieval database and full map bytes are charged.

## Prior-art delta

PA292's RANF retrieves acoustically similar listeners and conditions a neural field; its open implementation also includes learning-free nearest-neighbor and HRTF-selection controls. PA293 uses a direction-conditioned native listener latent. This experiment tests a compact sparse code around a shared data-derived field. The environment lacks PyTorch/GPU, so it does not reproduce or claim RANF neural quality. The nearest training-listener retrieval control is computed only from allowed sparse calibration measurements.

## Data and protocol

The measured data are SONICOM 48 kHz `FreeFieldCompMinPhase` SOFA files. Fixed split: train P0001–P0030, development P0031–P0036, fresh P0037–P0042. The split, supported direction indices, rank, methods, serializer, metrics and gates were fixed before any subject HRTF file was downloaded. Input file hashes and byte sizes are in `source/data_provenance.csv`; the downloaded SOFA files are excluded from Git.

The RANF challenge's fixed 3- and 5-direction sets were used. Quality is evaluated on all non-calibration directions with the official Spatial Audio Metrics Toolbox calculations for log-spectral distortion, interaural level difference error and interaural time difference error. The ITD helper from `spatialaudiometrics 0.0.8` has a Python-list division bug; the documented MAXIACC formula was reproduced with NumPy and checked in metric replay. No subjective listener test is claimed. The recorded wall times include shared code construction and metric evaluation; they are not isolated streaming inference latency.

## H / T / D / C / U

H: a shared HRTF field plus sparse listener code can recover useful per-listener variation while reducing amortized full-map storage; an ordinary sparse code may tie it.
T: 30 training listeners; 6 development listeners; 6 fresh listeners; 3/5 calibration directions; rank-8 shared PCA field; top-4 OMP sparse code; dense rank-8 PCA control; ordinary top-4 sparse control; nearest measured direction; nearest training-listener retrieval; and independent full-map upper control. No neural optimizer updates. Fresh reports contain 72 rows and were replayed exactly for all non-wall-clock metrics, payload byte counts and hashes.
D: **FAIL.** At 3 points, Mirror mean LSD was 7.976 dB versus retrieval 5.763 dB (+2.213 dB); at 5 points, 8.258 versus 5.903 (+2.356 dB). All 6 fresh listeners missed the preregistered +1 dB LSD margin at both support levels. ILD/ITD were closer, but do not repair the LSD miss. The ordinary top-4 sparse PCA control matched Mirror's payload hash and every acoustic metric exactly for all 12 listener/support pairs. Dense rank-8 PCA used 182 B/listener versus 227 B for the sparse code; its mean LSD was 7.737/8.060 dB versus Mirror's 7.976/8.258 dB, although ILD/ITD tradeoffs differed slightly.
C: direct nearest-listener retrieval is a stronger quality method. The shared PCA basis is also costly: 14,616,798 B plus 227 B per listener, versus 1,624,230 B per independent full map. At 6 targets, shared amortized cost is 1.50x independent. It crosses byte break-even at 10 targets and reaches 0.375x at 24 targets by accounting alone; quality was measured on only 6 fresh targets, so the 24-target storage point is an arithmetic projection, not a 24-listener quality result. For reference, per-target accounting is 2,436,360 B at N=6 (1.50x independent), 1,461,907 B at N=10 (0.90x), 609,260 B at N=24 (0.375x), and 304,744 B at N=48 (0.188x). Retrieval stores a 48,722,145 B training-map database plus a 132 B choice state.
U: neural RANF/LoRA training, larger held-out subject cohorts, subjective localization, head-pose streaming and transfer to new microphone/headphone chains.

## Fact / Interpretation / Hypothesis

FACT: six fresh listeners × two support sizes × six methods yielded 72 rows. Mirror top-4 had the same actual payload hash and outputs as ordinary top-4 sparse PCA for every paired row. It failed the retrieval-relative LSD threshold for 6/6 listeners at both support levels. The shared field is 14.6 MB and independent listener payload is 1.62 MB; listener identity and support metadata are included in the serialized states.
INTERPRETATION: a compact shared listener field is storage-efficient only after substantial amortization, but its spectral quality did not match simple nearest-listener retrieval in this screen. The sparse code is standard sparse PCA addressing, with no isolated Mirror-specific effect. Dense PCA also uses fewer personalization bytes and slightly lower mean LSD than top-4 Mirror, with small ILD/ITD tradeoffs.
HYPOTHESIS: listener field sharing may become useful with stronger field capacity or additional physical measurements, but any code must outperform ordinary sparse latent controls and retrieval at the same storage and calibration budget.
BOUNDARY: measured HRTF signal metrics only. RANF neural and subjective perception are not evaluated. The 24-target storage point is projected by serializer arithmetic; only six fresh target qualities were measured.
