# MA-742 — R-GCN basis coefficients factorized by relation attributes

Status: FAIL (scoped synthetic Mirror-specific claim); relation-code family pause triggered with MA-741

## Fact

On 32 relations spanning eight binary attribute triples, the six fit combinations trained a linear attribute-to-basis generator; the other two combinations were held out. Held-out full matrices use 12,564 B and have zero error. Free four-basis R-GCN relation coefficients use 8,881 B but held-out operator nRMSE `.5737` and Hits@1 `.4922`. The attribute generator/Mirror uses 8,813 B and gets nRMSE `8e-8`, Hits@1 `1.0`. Mirror saves 29.8% versus full matrices but only 0.8% versus free R-GCN coefficients. Native attribute generator and Mirror payload hashes, bytes, outputs and compute are exact matches.

## Interpretation

Explicit relation attributes let a learned coefficient generator compose the two unseen attribute combinations. This is a strong native function-generalization result. The incremental Mirror claim fails: ordinary attribute-conditioned factorization exactly captures the code mapping, and the small byte delta versus free coefficients does not reach the frozen 25% gate.

## Hypothesis result

**FAIL** for Mirror-specific value. Fresh/audit was not opened after exact native alias. Together with MA-741, this is the second consecutive relational relation-code experiment whose apparent Mirror function is standard low-rank/attribute coefficient factorization. Per the experiment stop rule, pause the relation-code family pending a non-reparameterization hypothesis.

## Frozen protocol and scope

PA188. Four shared 16x16 bases; 32 relation identities (four per attribute triple); 24 relations across six triples fit the generator; eight relations across two unseen triples are held out. Sixty-four synthetic entity vectors drive operator and link ranking. Relation attributes, bases, generator, entities and IDs are fully serialized. The generator is fit on training attribute combinations only. No real graph or trained R-GCN is used.

## H / T / D / C / U

- **H:** Attribute-compositional relation Views generalize basis coefficients to held-out relation types with lower bytes than free R-GCN coefficients and improve on a native attribute generator.
- **T:** Seed 74201; eight binary triples, six fit/two held out, four relation instances/triple, dimension 16, four bases; full, free R-GCN, native attribute generator, Mirror and universal controls.
- **D:** **FAIL** for Mirror-specific value. Attribute composition gives exact held-out operators and links, but Mirror exactly aliases the native generator and saves only .8% versus free coefficients.
- **C:** A standard linear attribute-conditioned coefficient generator explains the result completely.
- **U:** Real KG attributes, trained R-GCN, noisy/incomplete relation metadata and whether another functional coordinate can improve beyond native relation composition.
