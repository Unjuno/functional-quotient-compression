# MA-741 — R-GCN relation weights as Mirror views

Status: FAIL (scoped synthetic Mirror-specific claim)

## Fact

On 32 synthetic relations with 24 fitting and 8 held out, independent full matrices use 35,231 B. Free four-basis R-GCN coefficients use 9,074 B. The rank-2 relation-code Mirror bank uses 9,028 B and exactly matches the native rank-2 coefficient factorization payload. Both recover held-out relation operators at nRMSE 0 with Hits@1/MRR 1.0. Universal tying uses 5,533 B but held-out operator nRMSE is `.3793`, Hits@1 `.9727`, MRR `.9798`. Mirror saves 74.4% versus full matrices but only 0.5% versus free R-GCN coefficient state. Results include shared entities and complete bank metadata.

## Interpretation

Low-rank relation coordinates preserve unseen relation transforms in this constructed rank-2 family, but the direct native rank-2 factorization is the same model and quality. Its byte difference from free R-GCN coefficients is under one percent after charging the shared basis, entities and metadata. Link ranking is saturated for the structured operators, so operator nRMSE is the more discriminating metric here.

## Hypothesis result

**FAIL** for Mirror-specific value and for the >=25% savings over free R-GCN coefficients. Fresh/audit was not opened after exact native factorization alias.

## Frozen protocol and scope

PA188. Thirty-two relation matrices are generated from four shared random bases with rank-2 relation coordinates; 24 relation identities fit the shared code generator and 8 are held out. Sixty-four entity embeddings define synthetic link-ranking queries. Full matrices, free basis coefficients, the rank-2 code/generator, and a universal tied transform are serialized with entity state and relation metadata. Operator reconstruction compute is charged for basis models. This is an oracle algebraic graph screen, not trained R-GCN or real knowledge graph evidence.

## H / T / D / C / U

- **H:** Structured relation Views compress R-GCN relation weights while preserving held-out relation quality and outperforming free basis coefficients.
- **T:** Seed 74101, 32 relations, 24 fit/8 held out, dimension 16, four shared bases, rank-2 codes, 64 entity vectors; independent, free R-GCN, rank-2 native factor, Mirror and universal controls.
- **D:** **FAIL.** Held-out function and ranking quality are exact, but Mirror saves only 0.5% versus R-GCN coefficient state and exactly aliases native rank-2 factorization.
- **C:** The result is standard low-rank factorization of R-GCN coefficients; no Mirror-specific operator is present.
- **U:** Trained graph convolution, real KG metrics, relation side-information noise, few-shot held-out relation adaptation and broader rank spectra.
