# MA-319 — Tucker matrix-bank Mirror layer coefficients

Status: protocol frozen before development. Dedicated branch: `research/ma-319-tucker-matrix-bank-mirror-20261008`.

## H — hypothesis

Replacing per-layer free rank-2 Tucker coefficients with shared-center/shared-radius circular Mirror coordinates will reduce the complete serialized model package by at least 5% while keeping fresh character-level NLL within 0.02 nat/token of rank-2 Tucker.

## Mirror insertion and controls

The insertion point is the coefficient row for each layer in four repeated nanoGPT block-matrix banks: attention QKV, attention output, MLP expansion and MLP output. Every bank uses a common mean plus rank-2 matrix basis. Mirror rows are a shared 2D center plus shared radius times `[cos(angle_l), sin(angle_l)]`. Controls are full independent weights and free-coefficient Tucker banks at ranks 1, 2 and 4. Tucker's free coefficients are the direct PA35 control.

## Frozen protocol

See `PROTOCOL.json`. Four-layer nanoGPT character LM: 4 heads, width 64, context 64; 500 AdamW updates per model seed. Tiny Shakespeare data is fetched by pinned SHA-256 and split sequentially 80/10/10. Development seeds are 31901/31902. Fresh seeds 31911/31912/31913 and the final text split stay sealed until source and protocol are committed. All bytes include untouched tensors, banks, layer addresses, metadata and serialization headers.

## Prior-art delta

PA35 already establishes shared Tucker matrix banks with free layer coefficients. This experiment asks whether a compact structured layer address adds measured storage/quality value beyond those direct free coefficients on an actual causal language model.
