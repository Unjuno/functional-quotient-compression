# MA-371 — MatFormer granularity Mirror views

Status: **FAIL** (bounded nested-width MLP screen)

PA53/PA107 establish that nested Transformer FFN widths support elastic and Mix'n'Match models. This screen tests only whether a small per-width/per-layer correction improves the three nested granularities. Unseen layer-granularity combinations are reserved for MA-372.

## H — falsifiable hypothesis

Per-layer, per-width four-angle Givens views over one nested-width supernetwork improve held-out quality versus native nested sharing while retaining at most half the independent-model payload, and beat byte-near FiLM/LoRA controls.

## Mirror insertion

Insert a width-specific Givens coordinate on the first eight active channels after each hidden-layer transform (and before the classifier for the final representation). Three widths × two layer slots produce a small code bank; inactive coordinates are untouched.

## T — frozen protocol

Digits v1.8.0 (recorded source hash), 60/20/20 stratified per-world splits, 2-layer width-64 nested MLP, widths 16/32/64, sandwich-rule supernet and distillation, 700 updates, 120 code updates per width. Development worlds 37100/37101 select LR from {0.003,0.01}; fresh worlds 37110/37111/37112 stay sealed until selection. Compare vanilla nested, Mirror, same-size FiLM, rank-1 LoRA and independent width-specific networks. Actual complete torch.save payload bytes are authoritative.

## Gates

See `PROTOCOL.json`. PASS requires all fresh worlds to meet the quality/byte gate and the separate Mirror-specific margin. FAIL includes parity or domination by simpler controls.


## Results

Fresh mean accuracies over widths and three worlds were 94.85% nested, 95.28% Mirror, 95.09% FiLM, 95.06% rank-1 LoRA, and 95.68% independent. Mirror misses the frozen +1pp and <=50%-of-independent-bytes gates: it improves by 0.43pp and uses 40,063/61,229 = 65.4% of independent bytes. Equal-byte FiLM is close with less correction compute. **Decision: FAIL for Mirror-specific value.** Replay details and H/T/D/C/U are in `STATUS.md` and `VERIFICATION.json`.
