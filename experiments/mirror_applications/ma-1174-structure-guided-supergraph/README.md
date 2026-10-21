# MA-1174 — Structure intervention as a *learning* operator, not merely pruning

**Mirror insertion:** one shared 128-FP32 physical two-layer MLP core is reused through 4 fixed signed-permutation pages, 32 trainable edge address codes and hard per-edge ON/OFF masks to generate 32 logical MLPs across four nodes and two recurrent passes. Householder–tanh–inverse Householder is the Mirror transform; direct-tanh and affine with the identical learned address structure are mandatory matched controls.

## Question

When a shared physical object trains through many virtual edges, does cutting a currently active path and *then continuing training* produce better functions than (i) keeping all paths, (ii) removing randomly, (iii) removing smallest gates, (iv) choosing edges with disagreeing task gradients, or (v) training the same final sparse topology from initialization?

The claim is about **how an intervention changes optimization**, not a claim that the resulting sparse graph alone must be smaller than another design. We use a bounded graph and record all search compute. New model parts (node, view, loop search) are separate later research.

## Worlds

Two independently sampled teachers: an eight-edge Mirror-world oracle with hidden edges; and a two-layer unrestricted MLP with no designed alignment to the student's fixed views. Teacher weight access is prohibited in students. Train/dev/fresh/fresh-OOD splits use independent seeds, and fresh is opened only after all models and masks are frozen. Compare identical time and optimizer schedules.

## Controls & strict precommitted verdict

See `PROTOCOL.json`, written before any run. Main claim requires a 5% improvement on five of six fresh pairs over *both* random/smallest-gate pruning and same-topology-from-scratch. Strong same-address direct nonlinear and affine comparisons are mandatory. For information efficiency, compare deployed bytes and measured active MAC/CPU latency; compare search training FLOPs/updates separately. Capacity and real-LM claims are out of scope.

## Reproduction

`python -m unittest discover -s tests -v`; `PYTHONPATH=source python source/run.py train`; `PYTHONPATH=source python source/run.py audit`.
