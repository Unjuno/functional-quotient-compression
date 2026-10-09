# MA-111 — semantic-role Mirror embedding

Status: **PROMISING; the registered fresh quality/storage gate passed 3/3**  
Branch: `research/ma-111-semantic-role-embedding-20261007`  
Base commit: `6e0ed10a094f9a082a144a2d8d86f508fd4cbad6`  
Protocol freeze: `03c101d1c9f37d0322283d053036a797766230f6`; selected-LR freeze before fresh: `b9aca4126c348c02d16237c79f40f119fbac3998`

## H — falsifiable hypothesis

One shared physical token embedding table and next-token readout, plus one small learned Givens coordinate per semantic role, can preserve role-conditioned next-token distributions on unseen lexical fillers with lower actual inference payload than private role maps. Roles whose transformations are not in that Givens family should need richer or private maps.

## Prior-art delta

PA13, *Attention as Binding: A Vector-Symbolic Perspective on Transformer Reasoning* (arXiv:2512.14709), interprets queries/keys as role cues and values as fillers and proposes explicit binding/unbinding heads and hyperdimensional memory. Zhang and McCoy (arXiv:2608.29034) fit Tensor Product Encoders to subject/verb/object representations, reporting R² above .60 on sampled LLM representations and above .90 on embedding-model representations. This motivates the direct role-conditioned controls here; the paper analyzes and reconstructs representations rather than testing a small learned role-view code.

Embedding controls are also established. *Weight Tying Biases Token Embeddings Towards the Output Space* (arXiv:2603.26663) reports that tying input/output matrices can favor output-space geometry. *Kronecker Embeddings* (arXiv:2605.29459) replaces input embeddings with fixed byte-position encodings plus a learned projection, including a controlled three-seed nanoGPT comparison. *Breaking Token Into Concepts: Exploring Extreme Compression in Token Representation Via Compositional Shared Semantics* (arXiv:2509.17737) proposes Aggregate Semantic Grouping (ASG), composing static token representations from shared semantic concepts. These methods establish important embedding-space/storage tradeoffs, but do not evaluate a role-conditioned Givens input view against ordinary role adapters.

Controls include additive role vectors and FiLM, per-role rank-2 input and output LoRA, a full per-role output head, fixed VSA sign binding, and full per-role input maps. The output-head controls can express the linear effect of a Givens input view and are strong functional alternatives.

## T — protocol and execution

This controlled one-step next-symbol task uses 32 filler-token IDs, four explicit roles (agent, patient, instrument, location), 16D shared input embeddings, and an 8-way next-token readout. The shared embedding/readout are frozen after initialization and charged to every method. Train uses filler IDs 0–23; held-out transfer uses unseen IDs 24–31 with fresh Gaussian input noise.

In the aligned family, each role teacher is the same shared function after a Givens rotation of the first embedding-coordinate pair. In the independent family, each role uses a random orthogonal input transform. The objective is temperature-2 forward KL on 128 examples per role for 300 updates per role. Development seeds 11101/11102 selected LR .01 by the registered all-method/all-family/all-seed mean held-out KL (.00322458 versus .00333996 at .003). The fresh LR was frozen before access. Fresh seeds were 11111–11113.

The inference codec charges the shared embedding/readout, all role state, role IDs, tensor names, dtypes, shapes, and format metadata. Resume bytes include optimizer state. Throughput uses a 2048-example batch, 5 warm-ups, and 50 timed CPU forwards with one torch thread.

## D — result

**PROMISING.** The preregistered aligned quality/storage gate passed in all three fresh seeds:

| Fresh seed | Mirror NLL | Mirror KL | Top-1 agreement | ECE delta vs full map | Mirror/full-map payload |
|---|---:|---:|---:|---:|---:|
| 11111 | 2.070963 | 1.17e-9 | 1.000 | -3.31e-8 | 0.474x |
| 11112 | 2.074079 | 0 | 1.000 | +2.20e-8 | 0.474x |
| 11113 | 2.070692 | 4.86e-10 | 1.000 | +1.22e-6 | 0.474x |

Fresh aligned means and measured deployment tradeoffs:

| Method | Total payload | Incremental role state | Resume payload | Mean teacher KL | Top-1 agreement | Active-compute proxy | Train wall | Inference examples/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mirror Givens | 2,770 B | 60 B | 8,243 B | 5.53e-10 | 1.000 | 15.67M | 0.433s | 25.41M |
| Rank-2 output LoRA | 3,411 B | 699 B | 12,751 B | 3.85e-9 | 0.9995 | 20.28M | 0.348s | 42.44M |
| Full per-role output head | 4,319 B | 1,608 B | 12,723 B | 6.38e-10 | 0.9999 | 29.49M | 0.267s | 57.64M |
| Full per-role input map | 5,839 B | 3,129 B | 17,331 B | 7.86e-10 | 1.000 | 44.24M | 0.266s | 65.11M |
| Hard tie | 2,708 B | 3 B | 4,893 B | 2.83e-5 | 0.9685 | 0 | 0 | 71.96M |

Mirror used 18.8% fewer total inference bytes than rank-2 output LoRA and 52.6% fewer than full role maps. Its role increment was 60 B versus 699 B for output LoRA. The MAC proxy was 0.773x output LoRA, but Mirror's measured training wall was 1.25x and throughput was 0.599x output LoRA. The full role head and input map were faster still. Storage/estimated arithmetic improved while runtime regressed.

On independent role maps, Mirror did not transfer: mean held-out KL was .01237 and top-1 agreement .391. Rank-2 adapters and role FiLM also struggled; the trained full input maps reached .966 agreement, and the exact teacher was 1.0. The semantic-role function therefore needs private/richer state outside the planted shared-view orbit.

The exact-teacher upper reference serialized to 2,753 B, 17 B below the trained Mirror record, because the reference stores all angles as one packed tensor while this Mirror serializer emits one named tensor per role. It is an oracle, not a learnable student, but it shows the current per-angle record layout is not the best possible byte encoding. No general Mirror-specific claim follows from the byte comparison alone.

## Fact / interpretation / hypothesis

**Facts.** The registered fresh aligned gate passed 3/3. Actual payload was 2,770 B for Mirror, 3,411 B for rank-2 output LoRA, and 5,839 B for full role maps. Mirror top-1 agreement was 1.0 in each fresh aligned world. The independent-role family did not transfer. A repeated fresh run reproduced all 600 rows' quality, bytes, examples, updates, and compute-proxy metrics exactly; timing fields were excluded.

**Interpretation.** In this synthetic family, one shared embedding/readout plus three role angles recovers four logical role-conditioned embedding functions and transfers to unseen filler IDs. The direct output-LoRA control reaches similar quality with a larger adapter and faster runtime. This is a storage/estimated-compute versus wall-time frontier tradeoff for a teacher deliberately generated from the same Givens family, not a broad semantic embedding result.

**Hypothesis.** Packed role-coordinate serialization may reduce the 60 B incremental code further. Natural semantic roles, larger vocabularies, causal-LM NLL, and optimized kernels could alter both quality and runtime; each needs a separate registered experiment.

## C — strongest counter-hypothesis

A per-role rank-2 output-LoRA update represents the same Givens-induced change in the shared linear readout, matches the held-out function, and has substantially higher measured throughput. The exact teacher's packed angle vector is also 17 B smaller than the current separately named Mirror-angle records. The observed advantage is specific to compact storage under the planted orbit and this serializer.

## U — unresolved

- No natural language model, real semantic-role supervision, or standard next-token corpus was used.
- No near-convergence capacity frontier, high-vocabulary test, or byte-matched generic embedding basis was measured.
- The per-role Givens code's optimized packed serialization and production GPU kernel remain untested.
