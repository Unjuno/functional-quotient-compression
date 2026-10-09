# MA-547 — Sparse weight edit versus routed hidden-space edit locality

Status: SCREENING  
Branch: `research/ma-547-roselora-mirror-locality-20261009`  
Base commit: `52a11e2f79b626865a78274bbbd5dd682d428c00`  
Prior art: PA96 (ReFT/LoReFT), PA104 (RoseLoRA)

## H — Hypothesis

A key-routed residual-space edit vector can transfer a synthetic counterfactual fact to held-out paraphrases with higher efficacy/locality than sparse RoseLoRA at similar actual bytes, and may justify its 512D state over a gated scalar-logit control.

## T — Execution

Pending. The screen uses a pinned Pythia-70M model, 16 synthetic key/value relabelings, four support and four held-out paraphrases per key, and dev seeds 54701/54702. Amendment 1 replaces the failing BPE subsequence matcher with the frozen exact UTF-8 key matcher. Initial output is preserved under `results/pre_amendment_1/`; rerun dev before fresh access. The key router, target margin, controls and analytic edit constructions are fixed; no parameter learning or hyperparameter tuning.

## D — Decision

Pending.

## C — Strongest counter-hypothesis

A residual activation vector is an ordinary ReFT-style intervention, while an exact key gate plus a scalar logit bias may realize the desired local edit at far fewer bytes. Sparse RoseLoRA may also achieve the edit globally with less locality.

## U — Not established

Pending. This is a synthetic counterfactual edit mechanism screen, not a natural factual knowledge editing claim.
