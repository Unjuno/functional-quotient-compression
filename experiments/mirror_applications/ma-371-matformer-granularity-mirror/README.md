# MA-371 — MatFormer granularity Mirror views

Status: **PROTOCOL FROZEN BEFORE DEVELOPMENT**

PA53/PA107 establish that nested Transformer FFN widths support elastic and Mix'n'Match models. This screen tests only whether a small per-width/per-layer correction improves the three nested granularities. Unseen layer-granularity combinations are reserved for MA-372.

## H — falsifiable hypothesis

Per-layer, per-width four-angle Givens views over one nested-width supernetwork improve held-out quality versus native nested sharing while retaining at most half the independent-model payload, and beat byte-near FiLM/LoRA controls.

## Mirror insertion

Insert a width-specific Givens coordinate on the first eight active channels after each hidden-layer transform (and before the classifier for the final representation). Three widths × two layer slots produce a small code bank; inactive coordinates are untouched.

## T — frozen protocol

Digits v1.8.0 (recorded source hash), 60/20/20 stratified per-world splits, 2-layer width-64 nested MLP, widths 16/32/64, sandwich-rule supernet and distillation, 700 updates, 120 code updates per width. Development worlds 37100/37101 select LR from {0.003,0.01}; fresh worlds 37110/37111/37112 stay sealed until selection. Compare vanilla nested, Mirror, same-size FiLM, rank-1 LoRA and independent width-specific networks. Actual complete torch.save payload bytes are authoritative.

## Gates

See `PROTOCOL.json`. PASS requires all fresh worlds to meet the quality/byte gate and the separate Mirror-specific margin. FAIL includes parity or domination by simpler controls.
