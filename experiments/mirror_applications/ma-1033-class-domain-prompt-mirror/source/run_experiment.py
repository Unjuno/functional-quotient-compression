"""MA-1033: class x domain prompt composition on frozen CLIP."""
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import snapshot_download
from PIL import Image, ImageEnhance, ImageFilter
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPModel
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
MODEL = "openai/clip-vit-base-patch32"
DATASET = "uoft-cs/cifar10"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
DOMAINS = ["original", "grayscale", "brightness_0.65", "gaussian_blur_radius_1.0"]
NCTX, UPDATES, BATCH, LR = 2, 20, 4, 0.001
METHODS = ["shared_maple", "mirror_product", "additive_code", "independent_cell"]
torch.set_num_threads(4)


def domain_image(image, domain):
    im = image.convert("RGB")
    if domain == "grayscale":
        im = Image.merge("RGB", (im.convert("L"),) * 3)
    elif domain == "brightness_0.65":
        im = ImageEnhance.Brightness(im).enhance(0.65)
    elif domain == "gaussian_blur_radius_1.0":
        im = im.filter(ImageFilter.GaussianBlur(1.0))
    return im


def choose(ds, classes, n, seed, excluded=None):
    labels = np.asarray(ds["label"])
    rng = np.random.default_rng(seed)
    excluded = excluded or {}
    out = {}
    for c in classes:
        ids = np.flatnonzero(labels == c)
        rng.shuffle(ids)
        banned = set(map(int, excluded.get(str(c), [])))
        got = [int(i) for i in ids if int(i) not in banned][:n]
        if len(got) != n:
            raise RuntimeError(f"not enough disjoint CIFAR images for class {c}")
        out[str(c)] = got
    return out


def previous_exclusions(which):
    names = ["ma-1024-coop-prompt-mirror", "ma-1025-cocoop-instance-view", "ma-1026-maple-coupled-view"]
    key = "train_indices" if which == "train" else "audit_indices"
    excluded = {str(c): set() for c in range(10)}
    for name in names:
        path = ROOT.parent / name / "runs" / "development.json"
        d = json.loads(path.read_text())
        for c, ids in d[key].items():
            excluded[str(c)].update(map(int, ids))
    return {c: sorted(v) for c, v in excluded.items()}


def assets():
    path = pathlib.Path(snapshot_download(MODEL, local_files_only=True))
    model = CLIPModel.from_pretrained(path, local_files_only=True).eval()
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    processor = CLIPImageProcessor.from_pretrained(path, local_files_only=True)
    files = []
    for name in ["config.json", "model.safetensors", "pytorch_model.bin", "vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "preprocessor_config.json"]:
        f = path / name
        if f.exists():
            raw = f.read_bytes()
            files.append({"name": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    return model, tok, processor, files, sum(x["bytes"] for x in files)


def initial_context(model, tok):
    ids = tok("a photo of a cat.", return_tensors="pt").input_ids
    return model.text_model.embeddings.token_embedding(ids)[0, 1:1 + NCTX].detach().clone()


def rotate_pair(x, angle):
    c, s = torch.cos(angle), torch.sin(angle)
    a, b = x[0], x[1]
    return torch.stack((c * a - s * b, s * a + c * b), dim=0)


class PromptState(nn.Module):
    def __init__(self, init, method, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.method = method
        self.ctx = nn.Parameter(init.clone())
        self.visual_projection = nn.Linear(512, 768)
        if method == "mirror_product":
            self.class_factor = nn.Parameter(torch.full((len(CLASSES),), 0.1))
            self.domain_factor = nn.Parameter(torch.full((len(DOMAINS),), 0.1))
        elif method == "additive_code":
            self.class_factor = nn.Parameter(torch.zeros(len(CLASSES)))
            self.domain_factor = nn.Parameter(torch.zeros(len(DOMAINS)))
        elif method == "independent_cell":
            self.cell_ctx = nn.Parameter(init[None, None].expand(len(CLASSES), len(DOMAINS), -1, -1).clone())

    def class_contexts(self, domain):
        if self.method == "independent_cell":
            # A full cell bank supplies the upper reference. Withheld cells receive
            # negative-class gradients only; they never receive positive examples.
            return self.cell_ctx[:, domain]
        if self.method == "mirror_product":
            angles = self.class_factor * self.domain_factor[domain]
        elif self.method == "additive_code":
            angles = self.class_factor + self.domain_factor[domain]
        else:
            return self.ctx[None].expand(len(CLASSES), -1, -1)
        c, s = torch.cos(angles), torch.sin(angles)
        a = self.ctx[0][None, :]
        b = self.ctx[1][None, :]
        return torch.stack((c[:, None] * a - s[:, None] * b,
                            s[:, None] * a + c[:, None] * b), dim=1)

    def visual_context(self):
        return self.visual_projection(self.ctx)


def text_ids(tok, names):
    x = tok("x", add_special_tokens=False).input_ids
    if len(x) != 1:
        raise RuntimeError("CLIP tokenizer must encode x as one token")
    ids = []
    for name in names:
        suffix = tok(" " + name + ".", add_special_tokens=False).input_ids
        row = ([tok.bos_token_id] + x * NCTX + suffix + [tok.eos_token_id])[:77]
        ids.append(row + [tok.pad_token_id] * (77 - len(row)))
    return torch.tensor(ids, dtype=torch.long)


def encode_text(model, tok, contexts):
    ids = text_ids(tok, CLASSES)
    tm = model.text_model
    h = tm.embeddings(input_ids=ids).clone()
    h[:, 1:1 + NCTX] = contexts
    mask = _create_4d_causal_attention_mask((len(CLASSES), 77), h.dtype, h.device)
    for layer in tm.encoder.layers:
        h = layer(h, attention_mask=None, causal_attention_mask=mask, output_attentions=False)[0]
    h = tm.final_layer_norm(h)
    eos = (ids == tok.eos_token_id).int().argmax(-1)
    pooled = h[torch.arange(len(CLASSES)), eos]
    return F.normalize(model.text_projection(pooled), dim=-1)


def encode_image(model, pixels, visual_ctx):
    vm = model.vision_model
    h = vm.embeddings(pixel_values=pixels)
    h = torch.cat((h, visual_ctx[None].expand(len(pixels), -1, -1)), dim=1)
    h = vm.pre_layrnorm(h)
    for layer in vm.encoder.layers:
        h = layer(h, attention_mask=None, causal_attention_mask=None, output_attentions=False)[0]
    pooled = vm.post_layernorm(h[:, 0])
    return F.normalize(model.visual_projection(pooled), dim=-1)


def records(ds, ids):
    rows = []
    for c, xs in ids.items():
        for ix in xs:
            for d in range(len(DOMAINS)):
                if d == int(c) % len(DOMAINS):
                    continue
                rows.append((int(c), d, int(ix)))
    return rows


def train(method, model, tok, proc, ds, train_ids, seed):
    net = PromptState(initial_context(model, tok), method, seed)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    pool = records(ds, train_ids)
    rng = np.random.default_rng(seed)
    losses, examples = [], 0
    t0 = time.perf_counter()
    model.eval()
    for step in range(UPDATES):
        domain = step % len(DOMAINS)
        eligible = [r for r in pool if r[1] == domain]
        batch = [eligible[int(i)] for i in rng.choice(len(eligible), BATCH, replace=False)]
        ims = [domain_image(ds[ix]["img"], DOMAINS[d]) for c, d, ix in batch]
        px = proc(images=ims, return_tensors="pt").pixel_values
        ys = torch.tensor([c for c, d, ix in batch])
        tf = encode_text(model, tok, net.class_contexts(domain))
        vf = encode_image(model, px, net.visual_context())
        logits = model.logit_scale.exp().detach() * (vf @ tf.T)
        loss = F.cross_entropy(logits, ys)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        examples += BATCH
    return net, time.perf_counter() - t0, {"first_last_loss": [losses[0], losses[-1]], "examples": examples, "updates": UPDATES}


def payload(net, method):
    arrays = {k: v.detach().cpu().numpy().astype("<f4") for k, v in net.state_dict().items()}
    manifest = {"schema": "MA1033-self-describing-v1", "model": MODEL, "method": method,
                "classes": CLASSES, "domains": DOMAINS, "n_context": NCTX,
                "template": "x x {class}.", "holdout_rule": "domain_id=class_id mod 4",
                "coordinate": "a_c*b_d Givens angle" if method == "mirror_product" else method,
                "tensor_manifest": [{"name": k, "shape": list(v.shape), "dtype": "float32-le"} for k, v in arrays.items()]}
    return json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode() + b"\n" + b"".join(v.tobytes(order="C") for v in arrays.values())


def evaluate(method, net, model, tok, proc, ds, audit_ids):
    preds, labels, domains, elapsed = [], [], [], 0.0
    model.eval()
    with torch.inference_mode():
        for d, domain_name in enumerate(DOMAINS):
            tf = F.normalize(model.get_text_features(**tok([f"a photo of a {c}." for c in CLASSES], padding="max_length", max_length=77, truncation=True, return_tensors="pt")), dim=-1) if method == "zero_shot" else encode_text(model, tok, net.class_contexts(d))
            visual = None if method == "zero_shot" else net.visual_context()
            items = [(int(c), int(ix)) for c, arr in audit_ids.items() for ix in arr]
            for start in range(0, len(items), BATCH):
                batch = items[start:start + BATCH]
                ims = [domain_image(ds[ix]["img"], domain_name) for c, ix in batch]
                px = proc(images=ims, return_tensors="pt").pixel_values
                t0 = time.perf_counter()
                vf = F.normalize(model.get_image_features(pixel_values=px), dim=-1) if method == "zero_shot" else encode_image(model, px, visual)
                score = vf @ tf.T
                elapsed += time.perf_counter() - t0
                preds.extend(score.argmax(-1).tolist()); labels.extend([c for c, ix in batch]); domains.extend([d] * len(batch))
    p, y, dd = np.asarray(preds), np.asarray(labels), np.asarray(domains)
    held = np.array([d == c % len(DOMAINS) for c, d in zip(y, dd)])
    correct = p == y
    def acc(mask): return float(correct[mask].mean()) if mask.any() else float("nan")
    obs, hld = acc(~held), acc(held)
    hmean = 2 * obs * hld / (obs + hld) if np.isfinite(obs) and np.isfinite(hld) and obs + hld else float("nan")
    return {"observed_pair_accuracy": obs, "heldout_pair_accuracy": hld, "seen_heldout_hmean": hmean,
            "all_accuracy": float(correct.mean()), "per_domain_accuracy": {DOMAINS[d]: acc(dd == d) for d in range(len(DOMAINS))},
            "audit_images": int(len(y)), "encoder_wall_s": elapsed, "images_per_s": len(y) / max(elapsed, 1e-12)}


def run(lane, seed):
    RUNS.mkdir(exist_ok=True)
    model, tok, proc, files, backbone = assets()
    model.requires_grad_(False)
    for p in model.parameters(): p.requires_grad_(False)
    train_ds = load_dataset(DATASET, split="train")
    train_ids = choose(train_ds, range(10), 8, 103300, previous_exclusions("train"))
    nets, train_wall, train_meta = {}, {}, {}
    for method in METHODS:
        nets[method], train_wall[method], train_meta[method] = train(method, model, tok, proc, train_ds, train_ids, 103300)
    for method in METHODS:
        (RUNS / f"payload_{method}.bin").write_bytes(payload(nets[method], method))
    # Audit labels and pixels are first accessed after every state is frozen.
    test_ds = load_dataset(DATASET, split="test")
    audit_ids = choose(test_ds, range(10), 8, seed, previous_exclusions("test"))
    image_hashes = {c: [hashlib.sha256(test_ds[int(ix)]["img"].tobytes()).hexdigest() for ix in xs] for c, xs in audit_ids.items()}
    results = {"zero_shot": evaluate("zero_shot", None, model, tok, proc, test_ds, audit_ids)}
    results.update({m: evaluate(m, nets[m], model, tok, proc, test_ds, audit_ids) for m in METHODS})
    pbytes = {m: (RUNS / f"payload_{m}.bin").stat().st_size for m in METHODS}
    payloads = {m: (RUNS / f"payload_{m}.bin").read_bytes() for m in METHODS}
    # Zero-shot requires no learned arrays, but still pays class/domain/template metadata.
    z = json.dumps({"schema":"MA1033-self-describing-v1","model":MODEL,"method":"zero_shot","classes":CLASSES,"domains":DOMAINS,"template":"a photo of a {class}."},sort_keys=True,separators=(",",":")).encode()+b"\n"
    (RUNS / "payload_zero_shot.bin").write_bytes(z)
    pbytes["zero_shot"] = len(z); payloads["zero_shot"] = z
    doc = {"schema":"MA1033-v1","lane":lane,"seed":seed,"model":MODEL,"model_files":files,"backbone_bytes":backbone,
           "dataset":DATASET,"train_fingerprint":train_ds._fingerprint,"test_fingerprint":test_ds._fingerprint,
           "train_indices":train_ids,"audit_indices":audit_ids,"audit_image_sha256":image_hashes,
           "heldout_rule":"domain_id=class_id mod 4","fresh_opened_after_freeze":True,
           "train_meta":train_meta,"train_wall_s":train_wall,"results":results,
           "prompt_payload_bytes":pbytes,"full_bundle_bytes":{m:backbone+b for m,b in pbytes.items()},
           "payload_sha256":{m:hashlib.sha256(b).hexdigest() for m,b in payloads.items()},
           "code_sha256":hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
           "backbone_parameters":sum(p.numel() for p in model.parameters()),
           "total_wall_s_including_model_load":None,
           "boundaries":["CIFAR-10, synthetic image transforms, frozen CLIP ViT-B/32, shallow two-token prompt screen; not official CoOp/CoCoOp/MaPLe reproduction"]}
    (RUNS / f"{lane}_{seed}.json").write_text(json.dumps(doc,indent=2)+"\n")
    print(json.dumps(doc,indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", choices=["development", "audit"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    run(a.lane, a.seed)
