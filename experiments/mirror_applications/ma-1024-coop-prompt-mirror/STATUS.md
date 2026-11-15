# MA-1024 status

Stage: COMPLETE — FAIL (CIFAR-10/CLIP prompt screen).
Branch: `research/ma-1024-coop-prompt-mirror-20261115`.
Base evidence: MA-1017/quantum family diagnostic tip `64d11ac4c`.
Fresh official CIFAR-10 test split: opened after prompt freeze; not used for tuning.
H: domain Givens phase codes preserve base-to-novel accuracy with fewer prompt bytes than independent domain contexts.
T: frozen CLIP ViT-B/32; CIFAR-10 4-shot base classes; four transforms; 30 updates per method; shared, independent, Mirror and native scalar controls.
D: FAIL. Mirror prompt state saves 73.8% vs independent, but novel accuracy is 8.59 pp lower; harmonic mean misses the 1 pp tolerance. Shared CoOp uses 73 fewer bytes and generalizes better.
C: four-shot adaptation may underfit each domain.
U: other datasets/backbones, natural shifts, private residual frontier, CoCoOp/MaPLe and external seeds.
Last verified commit: `502fc4a1234b1fa9c5e7d202e717106b03798992`.
