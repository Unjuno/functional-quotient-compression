# MA-980 results — ideal photonic simulation screen

Status: **NOT ESTABLISHED** for the registered storage frontier. Ideal simulated quality and serialized results are retained, but the native-vs-Mirror byte comparison is confounded by unequal serialization key/metadata strings.

## H — hypothesis

Eight phase-related 4×4 unitary tasks can share a fixed Hadamard optical core plus small task Views, preserving quality with fewer bytes than independent dense operators. A native phase bank is the primary simpler control.

## T — execution

**Fact.** Three deterministic fresh worlds (98011, 98013, 98017), each with eight synthetic operators `D_out H D_in`, were evaluated using decoded `torch.save` payload state. Development world 98001 was separate. Controls were shared H, direct native phase bank, Mirror phase View, and independent dense complex matrices. Fixed random complex probe vectors and relative Frobenius error were evaluated after loading serialized state.

**Fact.** Across all 24 fresh tasks, direct phase bank, Mirror phase View, and independent dense operators had zero measured operator/probe error. Shared H had mean relative Frobenius error 1.4065 (world means: 1.4498, 1.4287, 1.3411). Mean serialized sizes were 2,913 B direct phase bank, 2,849 B Mirror, 3,753 B dense independent, and 1,961 B shared H.

**Fact.** The direct and Mirror state contain identical phase tensor values, shapes and dtypes, but the archive used different field names (`input_phase`/`output_phase` versus `m_input`/`m_output`) and different metadata strings (`native-phase-bank` versus `mirror-view`). The 64 B apparent Mirror saving is therefore caused by serialization representation, not a smaller coordinate or physical state. Fresh data were already inspected; this ID was not rerun after discovery.

## D — decision

**NOT ESTABLISHED.** The simulation confirms that phase codes reconstruct the deliberately phase-aligned synthetic operators exactly and use fewer bytes than dense 4×4 matrices. It does not establish any Mirror-specific byte gain over the equivalent native phase bank: the measured 64 B difference is an artifact of unequal key/metadata lengths. The preregistered <=50% dense-payload gate is also missed (Mirror is 75.9% of dense bytes). No physical hardware frontier is measured.

## C — strongest counter-hypothesis

**Interpretation.** The whole benefit is ordinary phase-factor parameterization already supplied by a native optical phase bank. Apparent Mirror-specific compression vanishes if both controls use equal serialization schemas.

## U — unresolved

- Matched-schema actual byte comparison under a new fresh test set.
- More representative photonic operator families and device constraints.
- Optical loss, phase quantization, calibration, programming bits, switching energy/latency, and physical footprint.
- Whether a distinct Mirror coordinate beats native MZI/LightPro programming.

## Evidence separation

- **Fact:** simulated errors, decoded payload sizes, exact target construction, and schema mismatch.
- **Interpretation:** native phase bank explains the entire useful functionality; byte advantage is not established.
- **Hypothesis:** structured phase Views may still reduce state relative to dense task matrices, but this is established only as an ideal synthetic factorization, not as a device-level Mirror result.
