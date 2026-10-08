# MA-392 — Factorized token × domain Mirror embedding address

Status: **FAIL — held-out quality/NLL gates missed; fresh seeds remain sealed.**
Evidence lane: QUALITY / STORAGE / HELD-OUT COMPOSITION / COMPUTE
Base commit: `f0e40b6`
Prior art: PA59, complementary compositional embeddings.

## H — hypothesis

A shared token-logit table plus one cyclic Fourier/Givens phase per domain should generalize to held-out token-domain pairs under cyclic domain shifts, approach a full per-domain map, and use fewer actual bytes. It should exceed additive and rank-4 adapter controls if those simpler forms cannot represent the phase shift compactly.

## Mirror insertion

The model shares one token table across domains and adds domain code `theta_d`. A real Fourier transform turns eight output coordinates into paired frequency components; one phase generates the frequency-scaled Givens rotations that shift class logits by the observed domain.

## T — protocol and execution

Protocol frozen before development: two seeds (39201/39202), 128 tokens, 8 domains/classes, 1,600 Adam updates, and a seeded 25% held-out token-domain pair set. The target is `(token mod 8 + domain) mod 8`. Each method sees the same training pairs and domain ID. Controls: independent per-domain token tables, additive domain bias, rank-4 domain adapter, full domain map, and Mirror phase. Test metrics are measured after FP16 payload reload.

## D — decision

**Fact:** Held-out accuracy for Mirror was 0.8692/0.8779, versus rank-4 adapter 0.9916/0.9962 and full per-domain map 1.000/1.000. Mirror held-out NLL was 0.991/1.043 versus rank-4 0.044/0.014 and domain map 0.00054/0.00086. Mirror payload was 2,749 B versus 4,121 B domain map (66.7%), so the byte gate passed. Additive bias was 0.0 held-out accuracy and independent per-domain embeddings were near chance on unseen pairs. Seen-pair Mirror accuracy was 0.9987/1.000, showing the gap is specific to held-out combinations. Ten payloads replayed seen/held-out accuracy and NLL exactly; four tests passed. Fresh seeds 39211–39213 remain unopened.

**Interpretation:** Mirror compresses the per-domain representation and fits seen pairs, but the single phase code does not generalize to withheld token-domain combinations as accurately or confidently as ordinary rank-4 adaptation. The preregistered quality and NLL gates failed in both seeds despite the storage gain.

**Hypothesis:** A per-domain phase is too restrictive when the shared token table has not learned a stable Fourier-aligned representation. The rank-4 adapter provides extra per-domain freedom that improves held-out accuracy at a larger payload.

## C — strongest counter-hypothesis

This is a small synthetic cyclic-shift task, and Mirror's code is designed for that algebra. Its held-out shortfall may be due to optimization under 1,600 fixed updates rather than a representation limit. The rank-4 control also approaches a full domain map at similar storage, making a claimed Mirror-specific advantage unsupported.

## U — unresolved

Natural domain adaptation, more domains, longer optimization, token frequency skew, learned or non-cyclic transforms, and efficient serving kernels are untested. No recommendation or language-model claim is made.
