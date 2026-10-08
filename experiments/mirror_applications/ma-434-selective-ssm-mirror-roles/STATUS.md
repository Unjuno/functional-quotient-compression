# MA-434 status

Status: **FAIL — verified development screen; fresh seeds sealed.**

## H
A compact role code shifting shared selective-state decay creates useful recurrent functions at lower bytes than full per-role SSM copies.

## T
Twelve roles; 16-state input-selective recurrence; 32 sequences/role; length 256; two development worlds; no training. Controls include one shared field, an exact native decay-bias alias, and independent full copies.

## D
The role-conditioned outputs matched the FP32 reference at NRMSE 0.000237/0.000212; one shared no-code SSM had 0.220/0.204. Role diversity RMS was 0.0689/0.0776. However, actual compressed payload was only 2,457 bytes versus 2,643 bytes for full per-role copies (0.930x), missing the <=0.80 byte gate. Native per-role log-decay bias used byte-identical payloads and exactly identical outputs. Fresh was not accessed.

## C
The role coordinate is exactly a native additive bias on the log-decay parameter. The NPZ compressor also captures the repeated B/C/selectivity weights in the nominal full-copy bank, so serialized-byte savings are much smaller than parameter-count savings.

## U
Full Mamba integration, training/learning efficiency, natural sequence tasks, low-bit formats, and larger role banks are untested. This does not establish language-model quality or capacity.

## Evidence separation
- **Fact:** native code and payload match exactly; replay of both development worlds passed; fresh untouched.
- **Interpretation:** role decay creates distinct synthetic dynamics, but not a Mirror-specific representation and not enough actual-byte reduction under the frozen gate.
- **Hypothesis:** other selective-SSM View operators may reduce bytes beyond ordinary decay-bias conditioning.
