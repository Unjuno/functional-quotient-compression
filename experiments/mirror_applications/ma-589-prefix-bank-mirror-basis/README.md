# MA-589 — Prefix bank compressed by a shared Mirror basis

Status: SCREENING  
Branch: `research/ma-589-prefix-bank-mirror-basis-20261009`  
Base commit: `f6bdb727`

## Hypothesis

**H:** A shared per-layer rank-128 latent basis plus per-context rank-4 residual View can preserve next-token quality across a bank of text-context KV prefixes while reducing actual serialized bytes versus storing each context cache independently.

**Insertion:** `m` is a per-token, per-context, per-layer, per-KV-role coefficient vector over the shared residual dictionary. Shared bases and all metadata are charged once; each context latent and code payload is charged separately.

PA116 motivates persistent prefix state. This protocol evaluates compression of actual text-derived KV prefix caches. It does not train Prefix-Tuning vectors, so any conclusion is limited to cache-state sharing.

## Controls and evaluation

Pythia-70M with WikiText-2: 32 train contexts fit bases; two validation seeds evaluate 16 independent 64-token context/query pairs each. Controls are FP16, shared rank-128 basis, Mirror residual codes, and the exact native shared-dictionary representation. Fresh test seeds stay sealed unless every development gate passes and the native control differs.
