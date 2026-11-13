# MA-983 results — synthetic CSI domain adaptation

Status: **FAIL for the preregistered Mirror-specific quality/storage frontier.** The Mirror coordinate is exactly the same function and serialized state as an ordinary matched domain embedding. It misses the independent-decoder quality gate and costs more bytes than the shared model.

## H — hypothesis

A low-dimensional persistent site/frequency coordinate on a shared CSI decoder improves unseen-domain reconstruction and beamforming at fixed latent feedback, using less persistent state than ordinary domain adapters.

## T — execution

**Fact.** The CPU screen used 8×8 complex Saleh-Valenzuela-inspired angular-delay channels, represented as 128 real values. A fixed orthonormal 128→16 encoder produced one shared 128-bit quantized latent per channel. The shared linear decoder was fit on four source domains. Each of four target domains per world had 64 calibration channels and 256 separately generated test channels. Fresh domain/path worlds were 98311, 98313, and 98317; development was 98301. Adaptation was fit on calibration only.

**Fact.** Controls were shared decoder without domain code, rank-1 Mirror residual basis plus scalar code, ordinary scalar domain embedding with the same basis/coordinate, group-FiLM, rank-1 LoRA decoder residual, and an independently fit per-target decoder. Every channel used 128 latent feedback bits plus a 2-bit target-domain ID. Persistent payloads were measured as actual decoded `torch.save` inference state. Test scores were computed from the loaded payload.

**Fact.** Fresh mean metrics were:

| Condition | NMSE (dB) | Rate (bits/s/Hz) | Persistent bytes | Feedback bits/channel |
|---|---:|---:|---:|---:|
| Shared, no domain code | -3.51 | 7.909 | 18,213 | 130 |
| Mirror rank-1 domain View | -3.89 | 7.985 | 26,781 | 130 |
| Ordinary matched embedding | -3.89 | 7.985 | 26,781 | 130 |
| Group-FiLM | -3.82 | 7.908 | 18,845 | 130 |
| Rank-1 LoRA | -3.97 | 8.027 | 20,893 | 130 |
| Independent target decoder | -21.06 | 8.848 | 51,233 | 130 |

**Fact.** Mirror reached 90.2% of independent-decoder mean spectral efficiency and missed the 95% gate. Its NMSE was more than 17 dB worse than independent. Mirror and ordinary embedding had identical serialized payload hashes, output metrics and 26,781 B payload in all three worlds. Mirror also added 8,568 B over the shared model. Wall time in `RESULTS_CORE.csv` covers serialization plus test scoring only; it excludes common source fitting and target calibration solves. Compute proxy is 2,048 multiply-accumulates/channel for the fixed latent decoder, with the same MAC count across decoder variants.

## D — decision

**FAIL.** The quality/storage gate is missed and the matched ordinary embedding reproduces the Mirror coordinate exactly. There is no Mirror-specific advantage in this synthetic screen.

## C — strongest counter-hypothesis

**Interpretation.** A rank-1 domain residual cannot span target channel shifts outside the source residual direction. The poor independent-vs-Mirror gap is a capacity result for this particular synthetic linear decoder. Group-FiLM does not recover it either, while a full target decoder does at higher bytes.

## U — unresolved

- Real COST2100/DeepMIMO or measured channels and the PA291 CsiNet implementation/checkpoint.
- Convolutional CsiNet behavior on angular-delay tensors.
- More Mirror basis ranks and private residual boundaries under a preregistered new experiment.
- Beamforming gain/feedback under standard channel normalization, coherence, multi-user interference, pilots, and RF energy.

## Evidence separation

- **Fact:** synthetic fresh metrics, payload hashes/bytes, feedback bits, calibration split sizes and controls.
- **Interpretation:** matched domain embedding exactly accounts for Mirror; rank-1 view lacks target capacity in this generated setup.
- **Hypothesis:** additional structured decoder directions or real-channel-trained CsiNet may change the frontier, but this experiment does not test that.
