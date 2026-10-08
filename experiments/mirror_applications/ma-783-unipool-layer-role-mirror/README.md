# MA-783 — UniPool global expert pool + Mirror layer role

Status: **FAIL (two-seed development screen; audit unopened)**
Branch: `research/ma-783-unipool-layer-role-mirror-20261008`
Baseline: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`
Draw 13 selected this ID uniformly from 555 eligible P0/UNTESTED rows. The exact ordered pool, seed, index and hash are in `source/`.

## H — Hypothesis

One global pool of four physical FFN experts plus eight per-layer Givens coordinates can recover enough layer-specific function to approach a layer-owned MoE on held-out character-language NLL at no more than 70% of its complete serialized inference bytes, and outperform byte-near FiLM and native depth-embedding controls.

## T — Execution

Frozen before corpus acquisition: four-layer, width-64 character Transformer; 1,200 AdamW updates per model; two world seeds (78301, 78302); six controls (dense, untied MoE, UniPool, UniPool+Givens, UniPool+FiLM, UniPool+depth embedding). PA205 NormRouter and aggregate shared-pool balancing were used. Development used fixed contiguous corpus bytes 0–80% for training and 80–90% for development. The final 10% audit span was not decoded or tokenized. No tuning occurred after the freeze.

Corpus SHA-256: `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed` (1,115,394 bytes). The full manifest is in `source/development_summary.json`; raw corpus and inference payloads are local ignored artifacts, with serialized sizes and SHA-256 values recorded there and in `RESULTS_CORE.csv`.

## D — FAIL

Both preregistered development gates failed, so the audit remained unopened.

- **Storage passed:** Givens payload was 612,501 bytes in each seed versus 1,423,166 bytes for untied MoE (0.430x; the gate was <=0.70x).
- **Quality/storage gate failed:** seed 78301 Givens NLL was 2.2463 versus untied 2.1151 (+0.1312 nats; limit +0.10). Seed 78302 was within the NLL margin (2.2451 vs 2.1908, +0.0543).
- **Mirror-specific gate failed:** the Givens model did not beat both byte-near controls by 0.02 nats in either seed. Its NLL was 2.2463 / 2.2451, compared with FiLM 2.2286 / 2.2471 and depth embedding 2.2316 / 2.2155.
- Givens payload was 612,501 bytes, FiLM 612,225, depth embedding 615,032, and UniPool 611,622. The shared-pool methods saved bytes relative to untied MoE, but Givens did not establish a Mirror-specific gain.
- Total measured training wall time was 915.9 seconds for all 12 runs on single-thread CPU. Per-run times and MAC proxies are in the CSV. This short fixed-budget mechanism screen is not a capacity result.

## C — Strongest counter-hypothesis

A single global pool provides most of the useful parameter sharing; ordinary FiLM or native depth conditioning recovers layer behavior as well as the Givens view. The observed NLL and payload results support that explanation at this screen's scope.

## U — Still unknown

Near-convergence capacity, additional natural language tasks, more fresh worlds, and optimized inference/runtime remain untested. The locked audit was not accessed because the frozen development gates failed. No conclusions extend beyond this small fixed-budget character-LM screen.

## Verification and provenance

`source/development_summary.json` records the gate decision and corpus manifest. `RESULTS_CORE.csv` contains all 12 development measurements. `VERIFICATION.json` records payload hashes, replay checks and audit status. The model adapter lives under `source/`; nanoGPT baseline files were not edited. The only tests were three preflight checks before corpus acquisition.
