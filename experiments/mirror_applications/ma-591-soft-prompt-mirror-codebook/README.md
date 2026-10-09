# MA-591 — Soft prompt Mirror codebook

Status: SCREENING  
Branch: `research/ma-591-soft-prompt-mirror-codebook-20261009`  
Base commit: `bfbb761d`

## Hypothesis

**H:** One shared rank-4 basis and small task coordinates can represent eight independently learned soft prompts at lower actual bytes while retaining held-out continuation quality.

PA117 freezes the language model and learns task-specific soft input embeddings. Here each distinct WikiText article defines one task; the prompt is trained on the first 1024 article tokens and evaluated on a disjoint suffix. Pythia weights stay frozen.

## Controls

Zero prompt, eight independent learned prompts, shared rank-4 prompt basis plus Mirror codes, and the exact native low-rank control. Every method uses the same eight virtual tokens at inference; report actual serialized prompt bytes and sequence/runtime overhead.
