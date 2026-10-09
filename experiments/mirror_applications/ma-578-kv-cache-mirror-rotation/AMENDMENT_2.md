# Amendment 2 — remove a quantizer-symmetry no-op

Before registered dev access, smoke outputs showed all rotation candidates had identical cache reconstruction error. The prior implementation applied signs/permutations after the Hadamard transform. For groupwise symmetric int4, those are exact quantizer symmetries: scales and reconstruction error cannot change. The transform order is corrected to apply signs/permutations before the Hadamard, producing genuinely distinct orthogonal views. A test now verifies different candidates produce distinct quantized reconstructions on anisotropic cache vectors. This is a pre-dev implementation correction; registered seeds/gates/splits and the cache objective are unchanged.

Prior source SHA-256: `78d78d52ad2ab8fbed537aa697a58183d7bd619cffd52718f30ed9fd0833afd4`
Amended source SHA-256: `bbc5a9724775404f4290516bd094a10bfdb4e589267e43b823eaa61418ae94ee`
Prior protocol SHA-256: `de100c29efdef3a3dbceafc51300cebba8d535d3839871756e76dc85aa35b407`
Amended protocol SHA-256: `59fb7e1f852e3235fc4de2a4c0424f228fdae4a39b8c532497cb5d58ef985943`
No registered dev or fresh text was accessed before this amendment.
