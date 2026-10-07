# Original first validation queue

This file preserves the original 25 P0 seed candidates. The registry has since expanded to 500 candidates and 159 P0 entries through literature research. **WORKER_QUEUE.md and STATUS_BOARD.md are operationally authoritative.** Priority may change from development evidence, never from fresh/audit evidence.

1. MA-003 — Mirror top-k expert
2. MA-005 — signed Mirror expert mixture
3. MA-009 — shared Mirror experts + rare private expert
4. MA-019 — shared expert basis + Mirror coefficient
5. MA-024 — one LoRA to many virtual LoRAs
6. MA-041 — one QKV to multiple Mirror heads
7. MA-048 — physical 4 heads to logical 16 heads
8. MA-061 — one KV head to many logical KV heads
9. MA-063 — Mirror-MQA
10. MA-076 — one block to multiple Mirror layers
11. MA-079 — learned layer address
12. MA-086 — Mirror layer groups
13. MA-111 — semantic-role Mirror embedding
14. MA-116 — Mirror-RoPE
15. MA-121 — packet phase-slot Mirror
16. MA-129 — Mirror verifier views
17. MA-156 — one quantized payload to many Mirror decodes
18. MA-160 — Mirror residual quantization
19. MA-171 — circular-convolution Mirror
20. MA-173 — FFT-phase Mirror
21. MA-181 — holographic expert address
22. MA-186 — new task via Mirror code only
23. MA-189 — freeze backbone + new Mirror
24. MA-199 — gradient-space Mirror
25. MA-208 — expert distillation into Mirror bank

## Execution order

Do not simply run 1-25 in numeric order. Start with families that share the nanoGPT modification point and can reuse one experiment harness:

A. FFN/expert/adapters: MA-003, 005, 009, 019, 024.
B. Attention/KV: MA-041, 048, 061, 063.
C. Depth/position: MA-076, 079, 086, 116.
D. Temporal: MA-121, 129.
E. Compression/binding: MA-156, 160, 171, 173, 181.
F. Continual/optimization/distillation: MA-186, 189, 199, 208.

Within each family: mechanism screen -> byte-near controls -> fresh replication -> only then combinations.
