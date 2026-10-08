# MA-397 — Product-address Mirror vocabulary with collision views

Status: **FAIL — native/hash gates pass, scalar margin misses the registered threshold.**
Evidence lane: QUALITY / STORAGE / COLLISION / CODE LENGTH / COMPUTE
Base commit: `bfb1692`
Prior art: PA43 Product Key Memory, PA58 Hash Embeddings, PA59 quotient/remainder embeddings.

## H — hypothesis

Four logical tokens share each of 1,024 discrete product-key addresses. One per-token Givens angle should resolve the four-way collisions with accuracy close to native two-weight Hash Embeddings, beat scalar gains by at least five points, and use at most 80% of the native importance-vector bytes.

## Mirror insertion

For token `t`, `q=t mod 32` and `r=floor(t/32) mod 32`. Pure product composition uses `A[q]+B[r]`. Mirror uses `cos(theta_t)A[q]+sin(theta_t)B[r]`. Each product address is paired with four tokens whose class labels differ, so performance must show useful collision resolution.

## T — protocol and execution

Protocol frozen before development: two seeds (39701/39702), vocabulary 4,096, 16D vectors, four classes, 1,200 Adam updates. Compared full table, pure product, scalar gain, native Hash Embeddings with two token coefficients, and Mirror angle. All methods use the same deterministic addresses and uniform token samples within each seed. Final metrics are measured from FP16-reloaded payloads, which include all weights, classifier, metadata and archive overhead.

## D — decision

**Fact:** Mirror reached 1.000 accuracy in all four collision groups and both seeds, matching native Hash Embeddings; full table also scored 1.000. Mirror used 11,874 B versus 20,076 B native hash (59.1%), passing the byte gate, and decoded 4,096 unique vectors. Pure product used only 3,446 B but scored 0.248/0.250 overall with 1,024 unique vectors. Scalar reached 0.9985 in seed 39701 and 0.9695 in seed 39702; Mirror's margins were 0.15 and 3.05 points, below the registered +5-point requirement. Mirror NLL was 0.0058/0.0166 versus scalar 0.1358/0.2141 and native hash 0.0040/0.0104. Thus the registered scalar margin gate failed despite strong collision/byte/NLL results. Ten serialized payloads replayed metrics and uniqueness exactly; four tests passed. Fresh seeds 39711–39713 remain sealed.

**Interpretation:** A one-angle-per-token code resolved collisions that the discrete product address alone could not, using about 41% fewer payload bytes than native two-coefficient Hash Embeddings. It also substantially improved NLL over the same-byte scalar control. However, scalar accuracy was already near ceiling in the first seed and within 3 points in the second, so the strict five-point accuracy gate was not met; the frozen result is FAIL. The error margin is due to the threshold, not a quality failure against native hash.

**Hypothesis:** The two component vectors plus one angle encode the four collision classes in this deliberately aligned task. Whether this structure helps under natural token distributions remains unknown.

## C — strongest counter-hypothesis

The target label is defined exactly by the four-way collision group. This is a designed mechanism test, not evidence that product-key collisions in real vocabularies align with useful semantic distinctions. Scalar gains also nearly solve the task, so the Mirror-specific accuracy advantage is small.

## U — unresolved

Natural vocabulary/LM quality, noisy or semantic collisions, larger component counts, frequency skew, serving kernels, and convergence capacity are untested.
