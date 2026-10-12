# CMW-002 pre-run protocol

Date 2026-10-12. Frozen source/test SHA256 4537251b545279ff663896b61341a8a64998d8800d2fee76f9532e7351639702, complete local protocol SHA256 e0971ab8f8d64ae404c6069a321849a9c844db815a1f73467da0cfe657aeb26e, local git commit a0300f7be779238afc7746a8ef49a46cf6cc170f.

Calibration seed 29000, 250 steps per 10 methods and 3 worlds, LR 0.01/0.03, select mean seen-validation nMSE. Development seed 29101-29103, 650 steps per method, 3 worlds (shared/partial_shift/unrelated). Reserved fresh 29201-29205 sealed. 4 objects, 12 action types, 12 directed pairs; 24 heldout kind-pair combinations, 2400 train and 700 validation samples. Compare mirror, mirror_gated, mirror_private, fixed_mirror, shared, native2, native2_gated, native_matrix, native_matrix_gated, native_mlp. Forward supervised with action; reverse inferred from observed before/after.

H1 state gate improves partial-shift nMSE over always-on private by >=20% in >=2 of 3 dev worlds; H2 shared damage <=0.01 nMSE; H3 nMSE <= native2_gated+0.005 AND complete NPZ bytes <=0.95 native2_gated in >=2 worlds; H4 144-class reverse event top1 >=70% shared and >=50% partial mean; H6 eager batch1/64 latency <=1.25 native2_gated. All required to open new fresh seeds. Native 2-coefficient basis is an exact functional alias in identity shared Mirror basis, hence no automatic Mirror novelty.
