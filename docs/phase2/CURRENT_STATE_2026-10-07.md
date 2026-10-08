# Phase II current state — 2026-10-07, through SRM003 and TM001

Status: **current navigation and interpretation, not a preregistration**.

## Executive conclusion

The hypothesis is a canonical shared backbone plus multiple reusable specialist residuals and only-when-needed private capacity. Whole-model Mirrorization, ES and recurrent training loops are not the default.

**SRM003 was executed, but the scientific adoption gate failed.** A small causal decoder learned all atomic mappings but did not reliably compose them from endpoint-token loss. More residual experts and validation-based pruning did not resolve this. TM001 then tested temporal packetization directly: multiple future tokens can be emitted in one forward when the information determining the packet is already available, but factorized parallel slots fail when a packet-level latent is still unresolved.

## Mirror program invariant

The application program is centered on the extra low-description functional parameter `m`, not on any one transform family.

Operationally:

`F(x; theta) -> F(x; theta, m)`.

The current mandate is to insert and stress-test `m` across as many strong existing mechanisms as practical, with the native method retained as a control. Manifold/quotient discovery, expert merging, KAN bases, tangent spaces, fast states, and other newer research lanes are supporting tools for choosing where and how `m` should act; they do not supersede the Mirror parameter as the program's central experimental variable.

See [MIRROR_PARAMETER_INTEGRATION_DOCTRINE.md](MIRROR_PARAMETER_INTEGRATION_DOCTRINE.md).

## What the latest evidence actually says

### SRM001: decomposable synthetic tasks

Multiple selected components had a strong fixed-update advantage in the count task and weaker signals on bitmask tasks. LM-loss-only routing still received explicitly structured rule-token inputs. The result is not unconstrained rule discovery in language. Some reported retention counters aggregate compound examples containing a rule; they must not be substituted for exhaustive atomic truth-table retention. Standard MoE improved with longer training, so near-convergence capacity superiority was not established.

### SRM002: ordered operators with a supplied executor

A teacher generated from a shared signed basis plus sparse rank-2 private residuals could be learned and compacted efficiently. Mirror stretch/shear was weaker than a signed shared basis on that family. Learned residual scores separated teacher-private slots in that controlled setting.

The operator fixture explicitly passed the updated state through the supplied ordered list of operators. It demonstrated learning reusable operators **with an executor**, not discovering an executor from a conventional decoder's final loss. Failure of a narrow shared model on another discrete family does not prove that family is intrinsically unshareable.

### SRM003: no oracle router or private mask

Two-layer causal Transformer, width32, 24 operators on16 states, explicit operator tokens, final answer CE. Three fresh world/init pairs; no auxiliary intermediate labels. Models train on both atomic and two-rule examples; pair combinations are held out with reverse-closed splits.

- All five main architectures retained all24 atomic rules over every one of16 inputs at1600 updates.
- Unseen-pair accuracy was only about9–16% across the five methods at1600 updates.
- Dense, full-MoE and Hybrid long controls reached about13–19% at6400 updates; they still retained all atomic rules.
- Hybrid did not satisfy the predeclared2-point advantage over both full-MoE and low-rank-MoE in every fresh setting.
- Two externally orchestrated calls using the model's **predicted**, not true, intermediate state scored100% in all9 long models and their1600-update parents. This supplies execution order and costs extra compute; it is not a one-forward success.
- Validation-pruning reduced Hybrid inference bytes90,329 ->85,672 (5.16%). It passed a numerical1-point tolerance at low composition accuracy, but was not consistently better than random pruning or small-from-start.
- Removing atomic examples gave zero exactly retained atomic rules in the pair-only diagnostic; query-position exposure also changed, so this is not a single-factor causal proof.

Primary comparisons controlled data/update opportunities, not total FLOPs or wall time. No LLM or general capacity claim is made.

### TM001: parallel period token generation

Two-layer width32 causal Transformer, 32 states, 12 transition rules, P=2/4/8.

- P=4 Direct and triangular-Mixer period decoders reached 100% joint accuracy in 3/3 deterministic fresh worlds.
- When the packet branch was random but exposed in context, Direct reached 100% in 3/3; Mixer reached 100%, 100%, 99.87%.
- When that one packet-level branch bit was hidden, P=4 median sequence NLL was 0.769 nat for KV-cached AR versus 3.318 Direct and 3.354 Mixer; AR generated valid trajectories in 100%, period decoders about 39–41%.
- On world201 hidden P=8, AR sequence NLL stayed 0.777 nat while Direct/Mixer rose to 10.49/10.77 nat and valid trajectories fell to 2.9%.
- Triangular slot mixing gave no stable quality advantage over independent phase slots.
- CPU one-thread batch1 throughput was about 3.1x cached AR for P=4 and 5.1–5.4x for P=8; speedup shrank with batch and P=2 batch128 was slower than AR.
- A shared packet-latent diagnostic improved the hidden case but did not reliably close the AR gap.

Interpretation: period-token parallelism is viable for conditionally determined packets. It is not a substitute for modeling joint uncertainty that is resolved inside the packet. This is a synthetic CPU mechanism result, not a natural-language or GPU claim.

## Mirror application exploration lane

A new explicit exploration lane treats the Mirror/View coordinate as a reusable design freedom rather than one fixed architecture. The question is whether an existing physically repeated object can be replaced by one shared object plus low-description addresses while retaining useful logical multiplicity.

The current registry contains **995 MA-xxx candidates** across MoE experts, LoRA/adapters, attention heads, KV/GQA, depth tying, FFNs, embeddings/position, packet decoding, memory/retrieval, quantization, holographic binding, continual learning/optimization, ensembles/distillation, SSM/runtime mechanisms, structured transforms, model merging, neural operators, relational graph models, diffusion control, neural cellular automata, invertible activation/flow views, global/reused expert pools, compositional latent dynamics, robot/action policies, matrix memories, programmable neural graphs, KAN edge functions, tangent/information geometry, causal engram memory, plastic/fast state, learned model manifolds, collective inference protocols, cross-model KV cache translators, multi-scene neural fields/4D Gaussian assets, speaker-adaptive TTS and audio codecs, generative flow maps/solvers, model stitching, video-chunk/frame reconstruction, learned equivariance, spiking threshold/time modulation, photonic physical operators, wireless beam/CSI/RIS configurations, and personalized HRTF spatial audio.

The eleventh 2026-10-08 literature sweep added MA-876..935 and PA236..265; all 60 new hypotheses are UNTESTED. Previously consolidated 47 MA evidence items remain 29 PROMISING and 18 FAIL. The next planned worker is still MA-255.

The twelfth 2026-10-08 research sweep added MA-936..995 and PA266..295 (60 further UNTESTED candidates, 47 P0 and 13 P1). The registry now contains 995 candidates, including 948 UNTESTED. No new model runs were executed as part of the literature sweep, and MA-255 remains the canonical next worker candidate.

This registry is a hypothesis backlog, not evidence. Each candidate must use actual serialized bytes and the relevant simple control. Mirror-specific value requires beating a simpler non-Mirror shared/low-rank alternative.

A compact nanoGPT-derived baseline is stored under `third_party/nanoGPT/` for common A/B experiments. The original user-supplied archive SHA-256 and license provenance are recorded there.

## Interpretation and open hypotheses

Fact: reusable atomic mappings can exist without successful in-model ordered execution.

Interpretation: shared/private FFN parameterization alone does not impose the state-passing structure needed for non-commutative composition. The mechanism could involve optimization, representation of intermediate states, placement, routing or limited depth; the present diagnostic does not uniquely identify one cause.

Hypothesis: a fixed-depth, non-recurrent composition interface may allow learned rules to be reused internally. TM001 shows that P output slots can be evaluated in one forward, but also shows that unresolved within-packet uncertainty needs a shared packet-level latent or another joint mechanism. The combined interface has not yet been demonstrated.

## Active evidence map

| Line | Role | Scope/status |
|---|---|---|
| Phase I/FQC | post-training compression | preserved; no general sharing win |
| MN/MT/RF/reachability | views, geometry and routing | count-only weak; robustness and analytical diagnostics |
| MS008–MS014 | sensor/world decomposition | functional transfer signals; quality-equivalent compression not reached |
| SRM001 | shared-rule synthetic causal tasks | controlled fixed-update signals; parity failed |
| SRM002 | operator composition and shared/private pruning | controlled factorized teacher with supplied execution |
| SRM003 | causal discovery and pruning without oracle routing | completed negative gate; storage and execution separated |
| TM001 | parallel period token generation | conditionally determined packets PASS; hidden packet latent boundary; CPU speed signal |

## Decision rules

Keep learning efficiency, observed retention, mathematical capacity, rule reuse and compression separate. Require actual serialized bytes, strong Dense/top-k MoE/low-rank controls, and compute frontiers for any performance claim. A longer unsuccessful training run is not a certified upper capacity bound. Mirror-specific value additionally needs a non-Mirror shared-rule comparison.

Primary scientific adoption gate: **FAIL**. Pruning hardware/byte mechanics: verified. Useful-quality compression: **not established**. Natural-language evidence: **not tested**.

## Navigation

- [Mirror application design space](MIRROR_APPLICATION_DESIGN_SPACE.md)
- [Mirror application prior-art map](MIRROR_APPLICATION_PRIOR_ART.md)
- [Mirror KV cache reuse design](MIRROR_KV_CACHE_REUSE.md)
- [Mirror application research notes through 2026-10-07](MIRROR_APPLICATION_RESEARCH_NOTES_2026-10-07.md)
- [Research notes 2026-10-08: cross-model cache, dynamic scenes, speech, generators, stitching](MIRROR_APPLICATION_RESEARCH_NOTES_2026-10-08.md)
- [Twelfth sweep: video, equivariance, spiking, photonic, wireless and HRTF research](MIRROR_APPLICATION_RESEARCH_NOTES_2026-10-08_TWELFTH_SWEEP.md)
- [Autonomous worker goal](../../GOAL.md)
- [Mirror application roadmap](../../roadmap/MIRROR_APPLICATION_ROADMAP.md)
- [TM001 report](TM001_PARALLEL_PERIOD_TOKEN_MIXING.md)
- [SRM003 report](SRM003_CAUSAL_DISCOVERY.md)
- [SRM003 runnable source, tests and counts](../../experiments/shared_rule_moe/srm003_20261007/README.md)
- [SRM002 original record](SRM002_NONCOMMUTATIVE_COMPOSITION.md)
- [SRM001 original record](SRM001_SHARED_RULE_MOE.md)
- [Architecture hypothesis](ARCHITECTURE_SPARSE_SHARED_RULE_MOE.md)
- [Experiment registry](EXPERIMENT_REGISTRY.md)
- [Roadmap](../../roadmap/PHASE2_SPARSE_RULE_ROADMAP.md)

Historical records are preserved. This document corrects earlier overbroad interpretations without replacing their original measurements or protocols.
