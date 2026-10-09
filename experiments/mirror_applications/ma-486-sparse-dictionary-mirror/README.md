# MA-486 — Sparse dictionary Mirror function representation

Status: **FAIL** for preregistered top-4 quality and bytes gates.

## H / T / D / C / U

**H:** Sparse addresses over a paid shared functional dictionary can reconstruct 4-sparse function vectors below half the bytes of independent dense functions at useful quality.

**T:** Synthetic 32D vectors generated from 4-sparse combinations of a fixed shared 64-atom dictionary. Compared independent dense FP32 functions, dense least-squares coefficients, signed sparse support, and sparse FP32 coefficients for top-k={2,4,8,16}. The 64x32 dictionary is charged in every shared method. Development worlds 48600-48601 explored top-k; fresh worlds 48610-48612 × seeds 0-2.

**D:** FAIL. At top-4, sparse Mirror averaged 12,769B (70.8% of dense) but NRMSE .1093, above .05. At top-8, error fell to .0258 but payload rose to 15,329B (85.0% of dense). Dense least-squares coefficients gave near-exact reconstruction but were larger (26,405B). Signed-only values were smaller but poor quality. No tested point met both preregistered quality and <=50% byte conditions.

**C:** The paid dictionary is a substantial fixed cost at only 128 functions; sparse support also has index overhead. This hand-built dictionary is not a learned neural functional basis.

**U:** Learned dictionary amortization at larger banks, entropy-coded supports, task utility beyond vector reconstruction, and neural-network function banks remain untested.
