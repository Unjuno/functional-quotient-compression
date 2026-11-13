# Upstream evidence

- PA253 NanoVoice: https://arxiv.org/abs/2409.15760; no runnable model/source found in prior MA-906/907/916 searches.
- PA254 HyperTTS: https://github.com/declare-lab/HyperTTS. Inspected local clone `/tmp/ma908-hypertts`; only located published checkpoint `/tmp/ma908-600000.pth.tar` is the README's LTS100 single-speaker pretrained model.
- HyperTTS README VCTK finetune checkpoint statements are commented out. VCTK configuration requires a multispeaker model, GE2E speaker embedding preprocessing and a 900k-step schedule.
- No checkpoint or temporary audio is committed.
