# MA-864 results

## Fact

- A pooled base MLP was trained on six source rotations (0°,45°,90°,135°,180°,225°); six source-task fine-tunes supplied a delta SVD basis. A 270° target was adapted from 32 train support examples for 100 updates. Development seed 73 selected base LR .1; method-specific rates were locked before fresh seeds 311, 331, 347.
- Heldout accuracy by seed:
  - Rank-3 shared code: .3889, .4083, .3167.
  - Rank-6 shared code: .3472, .3361, .3750.
  - Random rank-3: .3583, .3944, .3083.
  - Rank-3 plus private rank-1 fc2 residual: .4611, .4750, .4778.
  - Full fine-tuning: .6250, .8028, .6111.
  - Unadapted base: .3639, .3972, .3583.
- The private residual improves over rank-3 by 6.7–16.1 pp and beats rank-6 by 10.3–13.9 pp, but remains 13.3–32.8 pp below full tuning.
- Actual payloads: rank-3 base+basis+code 41,301 B; rank-6 70,229 B; rank-3+private rank-1 41,997 B; full adapted model 11,933 B. The private method is 40.2% smaller than rank-6, but 3.52× the full model.
- Base/source training used 680/43,120 and 1,530/97,020 updates/examples. Support adaptation used 100 updates on 32 examples. Reconstruction proxy is 7,230 MACs for rank-3, plus 320 MACs for rank-1 private fc2 residual; query proxy 852,480 MACs. Per-seed runtime was 2.71–3.81 sec.
- Mirror/native coefficient outputs were identical; private payload reload max logit difference was zero. Development discovered and corrected a torch.save backing-storage overcharge before fresh data; test asserts rank-3 and rank-6 serialized payloads scale separately.

## Interpretation

A small private residual makes the failed shared rank-3 basis more useful and is more byte-efficient than growing that basis to rank 6 in this screen. It still does not approach full fine-tune quality, and its shared basis cost makes one-task storage over three times larger than a full checkpoint. Thus it does not improve the requested quality/storage frontier for this one-task setting.

The coefficient Mirror code remains an identity re-encoding of native coefficients, so the adaptation improvement is due to private LoRA capacity, not Mirror.

## H / T / D / C / U

- **H:** A private rank-1 residual repairs off-manifold new-task error more efficiently than simply growing the shared subspace.
- **T:** MLP 64→32→10; six source rotation tasks; 270° held-out task; 32 support labels; 100 updates; dev seed 73; fresh seeds 311/331/347; rank3, rank6, random-rank3, private-rank1, base, and full-tune controls. Serialized payloads, updates, examples, MAC proxies, and wall time measured.
- **D:** **FAIL** for the preregistered full quality/storage gate and Mirror-specific attribution. Private rank-1 gives a consistent improvement over low-rank controls, but remains 13.3–32.8 pp below full tuning and uses 41,997 B vs 11,933 B.
- **C:** The target task update is not represented by the low-dimensional source-task delta span; one low-rank residual cannot encode the new rotation's feature shift.
- **U:** Larger/private residual ranks, multiple target tasks amortizing a shared basis, larger support sets, natural datasets, and language models remain untested.
