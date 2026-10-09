# MA-576 results

## Fact

On both registered development seeds, identity int4 audit-row NRMSE was 0.09731254. The SmoothQuant-style channel scale alone scored 0.10337599; QuaRot residual-only with one shared code scored 0.09694560. SmoothQuant plus per-matrix best-of-16 residual rotation scored 0.09954078. The same combined method with random residual codes and the native sequential SmoothQuant+QuaRot control also scored 0.09954078. The residual-Mirror and native-control weight, channel-scale and rotation-code arrays are exactly equal.

The combined per-matrix method uses 4,440,974 B total (2,058 B rotation codes + 13,074 B channel scales); native sequential uses the same payload. SmoothQuant alone uses 4,438,916 B; QuaRot-only uses 4,427,196 B. Thus the combined method is 13,778 B (0.311%) larger than the better single component, though within the frozen 1% byte cap. Actual NPZ stats and hashes are in `ARTIFACT_PROVENANCE.json`.

**T:** two registered seeds on 12 pinned Pythia-70M attention-dense/MLP-up matrices, groupwise int4, 16 exact block-Hadamard/sign/permutation residual candidates, 20% calibration output rows and complementary audit rows. Controls: identity, scale-only, rotation-only, scale+global rotation, per-matrix residual rotation, random residual rotation and native sequential scale+rotation. This is a matrix reconstruction screen, not natural-text NLL/perplexity. Fresh 57611–13 remained sealed.

**D:** FAIL. Residual rotation improves scale-only by 3.71%, but is 2.68% worse than rotation-only, and it exactly aliases the native sequential control. Random residual rotation has equal measured quality and uses 704 B fewer.

**C:** the diagonal scale estimate is derived from weight-matrix calibration rows rather than activation statistics; this SmoothQuant-style proxy and ordinary QuaRot explain the observations. The combined method adds state without beating QuaRot alone.

**U:** true activation-calibrated SmoothQuant, language-model NLL/perplexity, end-to-end inference latency, and performance on larger models. The registered hypothesis fails on this frozen output-row screen; these broader claims are not established.
