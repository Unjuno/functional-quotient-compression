# MA-982 results — synthetic Type-II site subspace

Status: **FAIL for the preregistered Mirror-specific frontier.** Site-conditioned subspaces improved over a global basis, but Mirror exactly matched a native Type-II phase code. Its byte reduction against storing complete site subspaces was only 5.5%, and the total model-plus-feedback reduction was 2.9%, below the 20% gate.

## H — hypothesis

A shared base subspace plus compact persistent site coordinates can preserve beamforming utility while reducing total persistent-plus-online description versus storing a separate subspace per site.

## T — execution

**Fact.** A synthetic CPU model used four sites, an 8-element narrowband ULA, rank-2 channel subspaces, 128 held-out users per site, 10 dB SNR, a 0.12 off-subspace leakage component, 8-bit real/imaginary coefficient quantization, and eight calibration probes per site. Development world 98201 was separate from fresh worlds 98211, 98213 and 98217. Conditions were global shared subspace, shared base plus Mirror site phase, native phase-parametric Type-II, stored per-site Type-II bases, and full quantized CSI feedback. Payloads were serialized with `torch.save` and decoded before scoring.

**Fact.** Across fresh worlds, mean rate was 3.2036 bits/s/Hz for Mirror, 3.2036 for native phase-parametric Type-II, 3.2046 for stored per-site Type-II, and 3.2322 for full quantized CSI. The shared global subspace without site code reached 2.6142. Mirror retained 99.11% of full-CSI mean rate in every world. Mean relative channel NMSE was 0.0147 for Mirror/native and 0.0138 for stored per-site Type-II.

**Fact.** Actual persistent payloads were 2,273 B for both Mirror and native phase-parametric Type-II, 2,405 B for stored per-site bases, and 1,325 B for the full-CSI receiver metadata (channel vectors are sent online). Online feedback was 34 bits/user for all Type-II controls and 130 bits/user for full CSI. With 512 users/world, persistent-plus-online descriptions were 35,592 bits for Mirror/native, 36,648 for stored per-site Type-II, and 77,160 for full CSI. Mirror saved only 2.9% versus stored per-site Type-II on this total measure. Eight pilot probes/site were applied to all conditions and are not converted into bits.

**Fact.** Mirror and native phase-parametric Type-II decode the same base and site phases. They had identical rates, NMSE, feedback bits, operation proxies and serialized payload size in all fresh worlds.

## D — decision

**FAIL.** The quality threshold is met, but the required 20% total-description reduction against stored per-site Type-II is missed substantially (2.9%). The native parametric Type-II control exactly aliases the Mirror representation. The larger signaling reduction against full CSI belongs to ordinary Type-II subspace feedback, which is already prior art PA289.

## C — strongest counter-hypothesis

**Interpretation.** Site-specific low-rank feedback is the source of the large feedback reduction. The persistent phase coordinate compresses an aligned array geometry, but native steering-angle parameters provide the same behavior; state savings are too small relative to the online coefficients to pass the total-description gate.

## U — unresolved

- Public measured site-specific channel traces and PA289's actual probe/fingerprint algorithm.
- Multi-user interference, coherence-window signaling, pilot cost in resource elements, and channel mismatch over time.
- Whether non-phase-related site subspaces require private state and how that changes total signaling.
- Real RF energy, latency, and field generalization.

## Evidence separation

- **Fact:** listed synthetic rates, NMSE, serialized bytes, feedback bit accounting and exact native/Mirror alias.
- **Interpretation:** ordinary Type-II compression explains CSI feedback savings; Mirror offers no demonstrated gain over its native parametric equivalent here.
- **Hypothesis:** distinct site geometry may change the private-state boundary, but it needs measured channels and matched PA289 controls.
