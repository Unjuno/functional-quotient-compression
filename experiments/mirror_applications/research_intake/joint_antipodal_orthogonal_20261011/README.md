# MV-JOINT-001 — Jointly trained antipodal and orthogonal Mirror Views (2026-10-11)

**Scope:** One shared trainable MLP trunk; 4 jointly supervised binary task-Views. Three development seeds, CPU FP32, **42 trained model instances ×700 updates =29,400 updates**. This directly tests joint multi-view learning, not the earlier post-fit orbit reconstruction study. **No real MoE router, LLM benchmark, general semantic-inversion claim, or Mirror-native capacity win.**

## Core hypothesis and directly matched controls
With learned shared 2D logits z(x), create views at 0°,90°,180°,270° and jointly train their task losses. Compare one-label-only anchor, four-label fixed views, angle-learned views with geometry Gram penalties λ=0/0.05/0.5, native free rank2 (same function family as learned angle+scale), and four independent heads on **the identical shared trunk**.

Two task groups on the *same sklearn 8x8 handwritten-digit images*:
- complementary: even / digit≥5 / odd / digit<5.
- non-complementary: even / digit≥5 / prime-digit (2,3,5,7) / divisible-by-3 (0,3,6,9).

All methods train from identical trunk initialization and sampled minibatch indices per seed/family. Three development split seeds 101–103. Fresh seeds 211–213 **SEALED** after development gates H2/H3 failed. AdamW 700 updates, batch128, lr 0.003, weight decay 0.0001, CPU one thread, FP32 eager. Full state incl. trunk/addresses/metadata serialized to actual NPZ. Development quality means are *not* confidence intervals.

## Empirical results (3 development seed means)

| Case | Method | BCE nat/label ↓ | Mean accuracy ↑ | Full inference NPZ bytes |
|---|---|---:|---:|---:|
| paired | anchor only, 0/90/180/270 | .39760 | 72.78% | 19,992 |
| paired | **all four loss terms, fixed angle** | **.08124** | **97.64%** | **19,972** |
| paired | learn angles λ0 | .10573 | 97.73% | 20,806 |
| paired | learn angles λ.05 | .10869 | 97.64% | 20,818 |
| paired | learn angles λ.5 | .10647 | 97.69% | 20,814 |
| paired | native free rank2 | .09381 | 97.69% | 20,510 |
| paired | native shared trunk + four full heads | .08426 | 97.62% | 20,180 |
| unrelated | fixed angle jointly trained | .31772 | 79.35% | 19,972 |
| unrelated | learn angles λ0 | .20781 | 92.43% | 20,806 |
| unrelated | learn angles λ.05 | .26561 | 89.91% | 20,818 |
| unrelated | learn angles λ.5 | .29535 | 86.23% | 20,814 |
| unrelated | native free rank2 | .20297 | 91.11% | 20,510 |
| unrelated | native full four heads | **.06562** | **97.69%** | **20,180** |

**H1 PASS (bounded):** paired fixed-cardinal joint loss achieved lower BCE in all three development seeds than native independent four heads (differences −.00105/−.00219/−.00584) while saving **208 B =1.03%** in total NPZ. However, fixed cardinal signs are **exactly the same function as TWO ordinary native heads plus negated logits**, so this is not a new Mirror benefit. When only role0 was supervised, role0/role2 accuracy was 97.87% but unsupervised 90° roles were 47.69%. Meaning on the second axis came from the *joint task loss*.

**H2 FAIL:** on non-complementary tasks, an angular Gram penalty brought view geometry much closer to 0°/90°/180°/270° (mean Gram MSE 0.448→0.0946→0.00273) but *increased* BCE .20781→.26561→.29535. Paired differences in the unpaired case for λ.05 vs λ0: +.04040,+.04744,+.08556 nat/label. Angular distance is not task-functional distinction.

**H3 FAIL:** learned-angle View parameters are a polar-coordinate reparameterization of the native rank2 coefficient rows: a_k = ||a_k||[cos atan2(a_k2,a_k1), sin atan2(a_k2,a_k1)]. Learned-angle saved NPZs are larger (20,806B vs free native rank2 20,510B), not >=5% lower in both task families. No Mirror-specific Pareto or new degrees-of-freedom claim.

## Rigorous structural boundary

Without independent role bias, logits form Z U^T with Z shaped N×2 and U 4×2, so rank of output-logit matrix ≤2. Four directions do **not** mean four independent trainable functions. Fixed u2=−u0, u3=−u1 produces exactly sigmoid(-z)=1−sigmoid(z); this only corresponds to reverse *meaning* where teacher labels are binary complements. The native two-head representation is exact.

For unrelated tasks, fixed opposites force contradictions on samples with y0=y2 or y1=y3. Each contradictory pair contributes at least 2 ln 2 to the sum of its two BCE losses. Hence average four-task BCE has a lower bound (ln 2)/2 × (q02+q13), where q are respective incompatibility rates. In development seeds, this is ~0.277–0.279 nat/label; observed fixed BCE .318. Thus enforcing maximum opposition can be mathematically detrimental to incompatible label families.

## Reproducibility and evidence boundary
- **8/8 unit tests PASS**; **42/42 actual NPZ payloads** reloaded and SHA/length checked; four score metrics per payload (**168 checked scalar results**) replay with max abs error 0. Three complete full train-and-serialize replays reproduced bitwise-identical SHA256.
- Intel Xeon Platinum 8573C / 4-core cgroup quota / 5 visible logical CPUs / one Torch thread / no clock lock / Python3.13.5 / Torch2.10.0+cpu / sklearn1.8.0 / FP32. Not a speed benchmark or language quality test.
- Full source, protocol lock, tests, all 42 trained NPZ, logs, summary, verification and detailed **Japanese report** were generated in conversation archive `MIRROR_JOINT_VIEWS_2026-10-11_FULL_EVIDENCE.zip`; SHA256 `b5ab1cd91fa783e77f6120f854b14f242397549ecdbd59fd421268ab4174ac46`. Lightweight source/results ZIP SHA256 `f752a4706c78e3525afaa6c9e3bb0575882b2a7b3d0d6baae73c35665a883b24`. **These archives are conversation attachments, not GitHub-hosted source.** This branch records the results/index but does not claim the full source is pushed.
- Relevant prior art: [Householder Reflection Adaptation (NeurIPS 2024)](https://proceedings.nips.cc/paper_files/paper/2024/hash/cdd0640218a27e9e2c0e52e324e25db0-Abstract-Conference.html) and [Supervised Contrastive Learning (NeurIPS 2020)](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html).

**Research interpretation:** A view arrangement that matches an *actual learned task relation*, trained with a joint objective, can share function. Geometry alone cannot assign reverse meaning or ensure specialized experts. Next controlled test should learn View relationships from task loss/gradients, compare native rank2/shared heads plus MoE routing at exactly matched bytes, and hold out natural task families. No official MA status or main-branch modifications.
