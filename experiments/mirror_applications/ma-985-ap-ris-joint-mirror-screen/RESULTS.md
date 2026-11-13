# MA-985 results — joint AP×RIS factorization

Status: **FAIL** for the preregistered quality/storage frontier. The restricted joint product code missed both gates, and the ordinary native product implementation exactly reproduced Mirror.

## H — hypothesis

A shared rank-2 AP beam basis crossed with a rank-2 RIS phase basis can preserve weighted sum rate across fresh channel configurations using less serialized policy state than a full per-configuration joint AP/RIS table.

## T — execution

**Fact.** The ideal CPU MU-MISO simulator used four AP antennas, eight RIS elements, two single-antenna users and eight independently generated channels per world. The development world was 98501; fresh worlds were 98511, 98513 and 98517. Controls were full AP/full RIS standard WMMSE with alternating RIS gradient updates, rank-2 AP/full RIS, full AP/rank-2 RIS, combined rank-2 Mirror, and ordinary native AP×RIS product code. Every channel configuration was optimized directly from a fixed zero phase initialization; this is not learned-policy generalization.

**Fact.** AP WMMSE updates enforce total transmit power P≤1 by Lagrange-multiplier bisection. The RIS phase update uses 3 alternating outer rounds and 20 gradient steps each, retaining the best phase iterate. Fresh mean WSRs were 2.6888 full WMMSE, 2.1817 rank-2 AP/full RIS, 1.9025 full AP/rank-2 RIS, and 1.4709 for both combined Mirror and native product code. The ratio of aggregate means was 54.7%; per-world ratios ranged from 47.9% to 59.4%.

**Fact.** Actual decoded `torch.save` payloads were 2,725 B full, 2,721 B AP-rank-2/full-RIS, 2,785 B full-AP/RIS-rank-2, and 2,781 B for both Mirror and native product code. The combined factorization was 2.1% larger than full. Its native control had identical payload hashes, WSR and reconstructed controls in all fresh worlds. Maximum observed sum-power residual was 2.4×10⁻⁷. Total active-compute MAC proxy was 67,296 for full and 35,808 for combined Mirror in the first fresh world; this proxy and CPU wall time include offline per-configuration optimization and are not inference latency or measured energy.

## D — decision

**FAIL.** The combined code misses the ≥90% WSR gate (54.7%) and the ≤50% payload gate (it is larger than full). The native product-code control exactly aliases Mirror. Power limits and serialized replay passed.

## C — strongest counter-hypothesis

**Interpretation.** On these independently generated multipath configurations, rank-2 AP and rank-2 RIS profiles jointly remove too much adaptive capacity. The AP-only and RIS-only ablations each outperform the combined code, suggesting interaction loss rather than a single bottleneck. The full-control WMMSE reference uses only 3×20 outer RIS gradient steps and is not a guaranteed global optimum. Channels are known and optimized one configuration at a time; this measures per-configuration compressed policy storage, not held-out learned-site/user generalization. The very small eight-configuration payload is dominated by archive metadata, so byte amortization is not representative of a large policy bank.

## U — unresolved

- Whether larger banks of held-out user/site combinations amortize the shared-basis metadata.
- A native learned product codebook or PA290 model-driven GNN under the same channels.
- Multi-user 3GPP propagation, channel estimation errors, quantized phases and interference.
- Whether private residuals recover WSR without removing the storage/compute advantage.
- Hardware RIS/AP energy, switching latency and total control overhead.

## Evidence separation

- **Fact:** fresh WSR, power residual, serialized payload bytes/hashes, compute proxies and exact Mirror/native equivalence.
- **Interpretation:** this rank-2 product factorization is too restrictive and provides no Mirror-specific mechanism beyond a native product codebook.
- **Hypothesis:** larger channel-role banks with a private residual may recover joint WSR, but require a new protocol or candidate and a matched model-driven control.
