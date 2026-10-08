# MA-393 — Adaptive-capacity embedding plus rare-token Mirror views

Status: **FAIL — no rare-token quality gain over simpler compact controls; fresh seeds sealed.**
Evidence lane: QUALITY / STORAGE / FREQUENCY BANDS / COMPUTE
Base commit: `16b06fd`
Prior art: PA60, Adaptive Input Representations.

## H — hypothesis

Under Zipf sampling, retain full embeddings for frequent tokens and replace rare 8D rows with 4D latent codes decoded through a shared basis and one Mirror angle per token. The Mirror view should improve rare-token accuracy over same-byte uniform 12D embeddings and the same low-rank basis without a view, while preserving head quality and overall NLL at lower bytes than adaptive input embeddings.

## T — protocol and execution

Protocol frozen before development: two seeds (39301/39302), vocabulary 1,024, 32D classifier input, eight classes, 64/192/768 head/middle/tail tokens, Zipf exponent 1.1, and 1,200 Adam updates. All methods received the same sampled tokens and labels within a seed. Compared full 32D, adaptive 32/16/8D, uniform 12D, shared rank-4 tail basis, rank-4 plus scalar, and rank-4 plus Mirror angle. All reported quality is after FP16 payload reload.

## D — decision

**Fact:** All six methods reached 1.000 accuracy in head, middle, tail and overall bands for both seeds. Mirror payload was 20,669 B versus adaptive 24,539 B (84.2%); this met the <=85% storage gate. But tail-basis and tail-scalar controls also reached 1.000 accuracy and used fewer bytes (18,897 B and 20,667 B). Uniform12 used 26,125 B and also reached 1.000 accuracy. Mirror therefore failed the registered +0.05 tail-accuracy gates. Overall NLL for Mirror was 0.0010/0.0010, similar to or slightly above scalar (0.0010/0.0009); tail NLL was 0.0016/0.0016 versus scalar 0.0011/0.0011 and basis 0.0020/0.0018. Every token in every band appeared in the training sample despite Zipf weighting. Twelve serialized payloads replayed band accuracy/NLL exactly; four tests passed. Fresh seeds 39311–39313 remain unopened.

**Interpretation:** The storage reduction against the adaptive table did not translate into a Mirror-specific quality gain. A lower-cost shared rank-4 basis and a same-quality scalar control matched the Mirror tail accuracy, so the angle adds bytes and generation work without improving this task's frontier.

**Hypothesis:** The token labels (`token_id mod 8`) are too regular and the 65,536 training examples cover all tail tokens, so this protocol does not create a meaningful rare-token information shortage. That ceiling explains the failure to observe a gain but does not change the registered result.

## C — strongest counter-hypothesis

With every token observed and a trivial periodic label rule, the experiment measures representation overhead more than rare-token generalization. More realistic long-tail outcomes require natural language data and rare tokens that are genuinely under-observed.

## U — unresolved

Natural language-model perplexity, long-tail token coverage with unseen/near-unseen tail items, output/input tying, severe Zipf regimes, frequency-dependent convergence, and efficient inference kernels remain untested.
