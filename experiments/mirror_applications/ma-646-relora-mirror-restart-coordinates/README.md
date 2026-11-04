# MA-646 — ReLoRA restart coordinates

Status: SCREENING. This closed-form mechanism screen measures whether reusing a low-rank update basis across restarts retains ReLoRA's cumulative rank growth.

The target is a rank-16 matrix update composed of four orthogonal rank-4 phase updates. Compare independent ReLoRA factors per phase, one static rank-4 basis reused at every phase, a shared rank-16 basis dictionary with phase addresses, and a native block-ID decoder identical to the Mirror version.

Actual serialized inference payload bytes are authoritative. The phase-factor/restart representation is measured separately. No optimizer training or neural LM claim is made. See `PROTOCOL.json`.
