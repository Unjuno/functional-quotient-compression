# MA-369 — Once-for-All subnetwork Mirror correction

Status: **FAIL** (bounded OFA-style MLP screen)  
Evidence lane: MECHANISM / STORAGE / QUALITY / COMPUTE / RUNTIME  
Base commit: `ab95e5d` (cumulative evidence branch descended from worker-ready `c935a90`)  
Prior art: PA52 Once-for-All; PA51 universally slimmable networks.

## H — falsifiable hypothesis

A four-angle View fitted per width/depth subnet on top of one shared OFA-style slimmable MLP will recover weight-sharing interference on held-out digits at lower total inference payload than six independently trained subnetworks. A byte-matched four-value FiLM control may explain the gain; Mirror-specific value requires beating it on the frozen quality/byte gate.

## Mirror insertion

> **Mirror insertion:** this experiment adds four Givens angles `m_a` to the shared hidden activation immediately before the shared classifier so each width/depth subnet can express a small functional correction without storing a separate classifier or full subnet weights.

- Shared physical object: two nested 64-wide hidden layers and a shared ten-class head.
- Architecture choices: widths {16, 32, 64} × depths {1, 2}; six subnets.
- Native method: OFA-style sandwich-rule subnet sampling with in-place distillation. This is a small MLP mechanism screen, not a full Once-for-All reproduction.
- Mirror coordinate: four trainable Givens angles per architecture, fitted after the shared supernet is frozen.
- Controls: no correction, four-value diagonal FiLM, rank-1 output LoRA, and six independently trained subnetworks.

## Data and split

Use sklearn `load_digits` (1797 8×8 images; normalized pixels divided by 16), version 1.8.0, raw-array SHA-256 `faf2d98f61250c1cdc0b114323133d3c58da04f5c5d28bb78e4bde3c936a75b1`. Each world uses its own stratified 60/20/20 train/validation/test split. Development worlds 36900/36901 select the shared training learning rate. Fresh worlds 36910–36912 are untouched until settings are frozen.

## Gates

### PROMISING (bounded screen)

On every fresh world, Mirror must improve mean subnet accuracy by at least 1 percentage point over the uncorrected shared supernet, stay within 2 points of independent subnetworks, and use ≤50% of their complete serialized bytes. To establish Mirror-specific value, it must additionally either beat FiLM by ≥1 point at byte-near cost (within 15%) or save ≥10% actual bytes while staying within 0.5 point of FiLM.

### FAIL

Miss the quality/storage gate in at least two fresh worlds, or FiLM/LoRA matches or dominates Mirror at equal or lower full payload and active compute.

### NOT ESTABLISHED

The independent control cannot learn the task, serialization/metric replay fails, or the frozen split/protocol is violated.

## Boundaries

Small grayscale digit classification only. Fresh results select no settings. Fixed update counts are not capacity evidence. No device-specific latency claim or full OFA/published benchmark reproduction.


## Results and ruling

Fresh worlds 36910–36912 (learning rate 0.01 selected only on development worlds 36900/36901) produced mean accuracy 95.09% for Mirror, 95.00% for uncorrected shared OFA, 95.09% for FiLM, 95.52% for rank-1 LoRA, and 95.71% for independent subnetworks. Mirror's +0.09 percentage-point difference from shared misses the frozen +1 point gate. FiLM matches Mirror at the same complete payload size (40,191 bytes) and uses fewer correction MACs. Independent bank size is 98,185 bytes; this sharing reduction is attributable to OFA-style shared weights rather than Mirror.

**Decision: FAIL** for Mirror-specific value under this bounded screen. See `STATUS.md` for H/T/D/C/U and Fact/Interpretation/Hypothesis separation, and `VERIFICATION.json` for exact payload and metric replay.
