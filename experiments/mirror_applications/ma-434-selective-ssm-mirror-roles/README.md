# MA-434 — selective SSM role views

Status: **FAIL at the frozen development screen; fresh sealed.** PA73 reviewed.

## H — Hypothesis
One shared input-selective diagonal SSM plus a scalar role view over log-decay can reproduce multiple recurrent functions with lower actual bytes than storing each full SSM.

## T — Frozen mechanism screen
A 16-state Mamba-like recurrence uses token-dependent delta, B_t and C_t, while role code m shifts the shared log-decay vector. Twelve codes, 32 sequences/role, length 256. Compare one shared SSM, native role-conditioned decay bias, and independent full per-role copies. This is a fixed-parameter synthetic mechanism/storage/runtime screen, not a trained Mamba or language-model experiment. Actual compressed payload bytes include all role state and metadata. Fresh seeds 43411–43413 remain sealed.

See `PROTOCOL.json` for exact recurrence, seeds, metrics and gates.

## D — Outcome
Serialized-output NRMSE was **0.000237 / 0.000212** for role-conditioned decay; the no-role shared SSM scored **0.2204 / 0.2042**. Role diversity RMS was **0.0689 / 0.0776**, so the coordinate creates distinct synthetic recurrent functions. Throughput was **4.82M / 5.06M** tokens/s versus **5.00M / 5.56M** for the native control (0.964 / 0.911x), clearing the runtime ratio.

Actual compressed payload was **2,457 bytes** for role views and **2,643 bytes** for independent full per-role copies (0.930x; 7.0% saving), missing the frozen <=0.80 byte gate. The native per-role log-decay bias had the exact same serialized bytes, hash, and outputs. Metric and payload replay passed. There were no training updates; this is a synthetic recurrence mechanism check only.

## C — Strongest counter-hypothesis
The role coordinate is ordinary native conditioning on the SSM log-decay vector. NPZ compression also captures repeated parameters in the nominal full-copy bank, so parameter-count compression overstates the actual serialized saving.

## U — Still unconfirmed
This is not a full Mamba block or language model. Learning quality, natural sequence tasks, larger dimensions/role banks, and alternative selective-state views remain untested.

**Evidence labels:** recurrence scores, bytes, throughput and exact native alias are facts; FAIL follows the frozen byte and no-alias gates; benefit on realistic selective SSMs is a hypothesis.
