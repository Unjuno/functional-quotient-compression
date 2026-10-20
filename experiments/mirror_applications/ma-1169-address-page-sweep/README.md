# MA-1169 — Address-page capacity sweep (4/8/16), weight-space Mirror versus direct affine

**Mirror insertion:** a single learned 289-float tensor generates 75 logical 6→8→6 MLP edge weights. Two learned edge-specific address coordinates select among 4, 8, or 16 deterministic signed-permutation pages of the same shared 64-coefficient core, then an orthogonal Householder → tanh → inverse-Householder weight decoder optionally creates a nonlinear Mirror view.

## Question
Does expanding the selectable physical-to-virtual address bank improve held-out function fitting, and does the Mirror decoder produce extra utility beyond the identical trainable address mapping with a simpler, non-Mirror direct nonlinear decoder?

## Prior art / ancestry
Extension of MA-1168, not a new proof that generated weights are independent. Compare same-address affine as the primary matched control, native289 as a strong compact direct-MLP control. Parameter superposition and BatchEnsemble are conceptual priors; this is only a small synthetic mechanism screen.

## Frozen lanes and gates
- One 289-coordinate `nn.Parameter` for every generated-edge model: core 64, gates 75, address codes 150. All 75 edges × 3 loops active, no prune; 8250 generated weight values but only 289 independently trained scalars.
- 4/8/16 bank sizes, Mirror and matched affine for every size, and native289. The 4-page version equals MA-1168 AddressModel. For 8/16, extend page bank deterministically, preserving first four physical maps, with fixed angular extra-page logits from the existing two code coordinates.
- Positive control: teacher draws from 8-page Mirror family; negative transfer: unrelated 6→25→6 native teacher. Teacher parameter tensors remain secret during student training. Distinct train/dev/fresh/OOD inputs and predetermined seeds.
- Primary: fresh NMSE after dev-best selection. Also OOD NMSE, all actual inference-packet bytes, deterministic runtime buffer bytes, one forward pass B=1/B=64 latency, active MLP MAC and decoder address mixing work.
- Screen page-expansion benefit separately from Mirror-specific utility. Predetermine strict >=5% ratio and >=5/6 pair wins; never call a 400-update advantage 'capacity' or statistical replication.
- No hyperparameter selection by fresh results, no previous negative MA report alterations, no main branch changes.

## Falsifiers
Simpler affine decoder matches Mirror quality at each page count, additional pages increase routing/working-set cost without a quality gain, or native289 dominates fresh and runtime.

## Status
Protocol frozen; CPU execution pending. See PROTOCOL.json and STATUS.md.
