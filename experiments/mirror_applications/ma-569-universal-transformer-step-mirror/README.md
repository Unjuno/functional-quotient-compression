# MA-569 — Universal Transformer step Mirror code

Status: **FAIL for Mirror-specific value**  
Branch: `research/ma-569-universal-transformer-step-mirror-20261009`  
Base commit: `dc505c56`  
Prior art: PA123 Universal Transformer

## H

A compact step coordinate can make a recurrent shared block express distinct useful logical depth functions, including held-out steps, with lower bytes than untied depth while outperforming a direct step embedding/LoRA control.

## Frozen mechanism screen

An 8D shared linear recurrent block is applied for eight steps. At each step, the teacher applies a smooth step-indexed Givens view to the shared transition. Fit/encode on even steps and evaluate odd held-out steps over fixed Gaussian probes. Compare hard-tied block, Mirror angle coordinate, direct scalar step embedding, static per-step rank-1 residual, and untied matrices. Two seeds; actual NPZ bytes include all bases, codes and metadata. Report step-wise output nMSE, distinct functions, bytes, recurrent MACs and wall time. No optimizer updates; aligned recurrence mechanism only.

PASS requires held-out step nMSE ≤1e-4, ≥20% fewer bytes than untied matrices, and ≥10% fewer than direct time/LoRA controls. FAIL if direct step code matches Mirror. Fresh sealed on alias.

## H / T / D / C / U

- **H:** Step-indexed Mirror coordinates create useful logical depth from one recurrent physical block.
- **T:** 8-step, 8D recurrent operator bank, two development seeds; even-step training/even-addressed smooth code, odd-step held-out; 10 rows.
- **D:** FAIL for Mirror-specific value. Mirror/direct time-angle both 1207B and odd-step nMSE 0; untied 2510B/nMSE 0; hard tied 716B/.16679; rank-1 LoRA 1192B/.10727.
- **C:** Time embeddings and static per-step LoRA are direct controls.
- **U:** Universal Transformer training, adaptive halting, language or algorithmic tasks, GPU throughput.
