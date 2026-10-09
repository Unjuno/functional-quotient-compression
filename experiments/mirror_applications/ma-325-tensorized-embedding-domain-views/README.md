# MA-325 — Tensorized embedding domain Mirror views

Status: PROMISING, narrowly for aligned synthetic embedding domains. Prior art: PA34 Tensorized Embedding Layers.

## H

A shared TT embedding bank plus a small per-domain Mirror angle may replace independent embedding tables or domain cores. Compare actual bytes and token lookup quality against TT independent cores and generic PCA.

## T

Synthetic 256×256 domain tables represented by four TT matrix modes of size 4 and rank-2 bonds; 32 domains. Compare independent full tables, shared outer cores with independent middle cores, Mirror rotation of a shared middle core, and generic rank-2 PCA. Test aligned and independent domains. Development worlds 32500–32501; fresh 32510–32512; three seeds each. All payloads include metadata and are decoded from fp16 before measuring full-table and sampled-token lookup error.

## D

PROMISING narrowly. Fresh aligned Mirror uses 636 B / lookup NRMSE 5.78e-4 vs generic PCA 981 B / 4.20e-4 and independent TT cores 4,530 B / 4.20e-4. Full tables use 4.19 MB. Independent domains need private cores; Mirror NRMSE is 1.288.

## C

The aligned domains are generated from the same Givens core action; PCA may explain the compression. Independent domains may require private cores.

## U

No token modeling, NLL, pretrained vocabulary or transfer learning evidence.

Development: aligned domains yield Mirror 636 B / lookup NRMSE 5.65e-4, generic PCA 981 B / 3.97e-4, TT independent domain cores 4,530 B / 3.97e-4, and independent full tables 4.19 MB. Mirror is ~86% smaller than independent TT cores and ~35% smaller than PCA, with still-low lookup error. Independent domains require TT private cores: Mirror lookup NRMSE ~1.34. Fresh tests frozen settings.

## Fresh result / scope

The aligned synthetic embedding result replicated across fresh worlds. Mirror uses far fewer bytes than independent TT domain cores and full tables, with a nearby generic PCA control. Independent domains need private TT cores. No claim is made about language modeling or real vocabularies.
