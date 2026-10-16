#!/usr/bin/env python3
"""Gate 1: official HF Transformers GPT-Neo runtime parity vs offline engine.
Read-only. Uses official validation story 0 as probe text (parity only, no selection).
"""
import sys, json, argparse, hashlib
from pathlib import Path
import torch
sys.path.insert(0, "experiments/t282/code")
from engine import BPETokenizer, forward, load_checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", default="models/HF-28M")
ap.add_argument("--output", default="runs/manifest-001/hf_parity.json")
ap.add_argument("--probe-index", type=int, default=0)
args = ap.parse_args()
MODEL = Path(args.model_dir)
torch.set_num_threads(4)

raw = Path("data/official/TinyStories-valid.txt").read_text()
stories = [s.strip() for s in raw.split("<|endoftext|>") if s.strip()]
print("n_stories_primary:", len(stories))
probe_text = stories[args.probe_index]
print("probe_chars:", len(probe_text))

# --- offline engine path ---
cfg, state, raw_ckpt = load_checkpoint(MODEL, "cpu")
tok = BPETokenizer(MODEL)
ids_offline = tok.encode(probe_text)
print("offline_ids_len:", len(ids_offline))

# --- official HF path ---
from transformers import AutoModelForCausalLM, AutoTokenizer
hf_tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False, local_files_only=True)
hf_ids = hf_tok.encode(probe_text, return_tensors="pt")
print("hf_ids_len:", hf_ids.shape[1])
print("tokenizer_id_parity:", list(hf_ids[0].numpy()) == ids_offline)
print("decode_roundtrip:", tok.decode(ids_offline) == probe_text)

model = AutoModelForCausalLM.from_pretrained(
    MODEL, trust_remote_code=False, local_files_only=True,
    dtype=torch.float32)
model.to("cpu")
model.eval()
ids_t = torch.tensor([ids_offline])
with torch.inference_mode():
    z_off = forward(state, cfg, ids_t)
    z_hf = model(hf_ids).logits
print("shapes:", tuple(z_off.shape), tuple(z_hf.shape))
diff = (z_off.float() - z_hf.float()).abs()
den = z_off.float().abs().clamp_min(1e-6)
print("max_abs_err:", float(diff.max()))
print("max_rel_err:", float((diff / den).max()))
print("mean_abs_err:", float(diff.mean()))
ref_nll = torch.nn.functional.cross_entropy(
    z_off[0, :-1].float(), ids_t[0, 1:], reduction="mean").item()
print("engine_ref_NLL_story0:", ref_nll)
out = {"model_dir": str(MODEL), "probe_index": args.probe_index,
       "stories_primary": len(stories), "probe_chars": len(probe_text),
       "offline_ids_len": len(ids_offline),
       "tokenizer_id_parity": list(hf_ids[0].numpy()) == ids_offline,
       "max_abs_err": float(diff.max()), "max_rel_err": float((diff / den).max()),
       "mean_abs_err": float(diff.mean()),
       "engine_ref_NLL_story0": ref_nll,
       "torch": torch.__version__}
import transformers
out["transformers"] = transformers.__version__
Path(args.output).write_text(json.dumps(out, indent=2))
print("wrote", args.output)
