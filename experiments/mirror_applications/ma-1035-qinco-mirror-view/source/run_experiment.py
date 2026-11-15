"""Small QINCo-v1-structure residual-codebook screen for MA-1035."""
from __future__ import annotations
import argparse, hashlib, json, math, pathlib, sys, time
import numpy as np
import torch
from torch import nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from qinco_reference import QINCo, QINCoStep
from quantization import assign_to_codebook

D, K, M, L, H = 32, 16, 4, 1, 32
NTRAIN, NVAL, NDB, NQUERY = 1024, 256, 512, 128
UPDATES, BATCH, LR = 80, 64, 0.001
METHODS = ["rq_static", "qinco_native", "qinco_tied", "qinco_mirror_givens", "qinco_additive_rank1"]
torch.set_num_threads(4)


def make_world(seed):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 2.5, size=(16, D)).astype(np.float32)
    scales = rng.uniform(0.25, 1.25, size=(16, D)).astype(np.float32)
    def sample(n):
        labels = rng.integers(0, 16, size=n)
        x = centers[labels] + rng.normal(size=(n, D)).astype(np.float32) * scales[labels]
        return x.astype(np.float32)
    return {"train": sample(NTRAIN), "validation": sample(NVAL), "database": sample(NDB), "queries": sample(NQUERY),
            "generation": {"seed": seed, "centers_sha256": hashlib.sha256(centers.tobytes()).hexdigest(),
                           "scales_sha256": hashlib.sha256(scales.tobytes()).hexdigest()}}


def kmeans(x, k, seed, iterations=12):
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(len(x), k, replace=False)].copy()
    for _ in range(iterations):
        dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        label = dist.argmin(1)
        for j in range(k):
            rows = x[label == j]
            centers[j] = rows.mean(0) if len(rows) else x[rng.integers(len(x))]
    return centers.astype(np.float32)


def initialize_centroids(x, seed):
    residual = torch.from_numpy(x.copy())
    codebooks = []
    for stage in range(M):
        centers = kmeans(residual.numpy(), K, seed + stage * 101)
        c = torch.from_numpy(centers)
        codes = assign_to_codebook(residual, c)
        codebooks.append(c)
        residual = residual - c[codes]
    return codebooks


def givens_view(x, angle):
    pairs = x.reshape(*x.shape[:-1], D // 2, 2)
    a, b = pairs[..., 0], pairs[..., 1]
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack((c * a - s * b, s * a + c * b), dim=-1).reshape_as(x)


class SharedStageQINCo(nn.Module):
    """One official-structure QINCoStep, reused with a stage-conditioned prefix view."""
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.codebook0 = nn.Embedding(K, D)
        self.step = QINCoStep(D, K, L, H)
        if mode == "qinco_mirror_givens":
            self.stage_angle = nn.Parameter(torch.full((M - 1,), 0.02))
        elif mode == "qinco_additive_rank1":
            self.stage_scale = nn.Parameter(torch.zeros(M - 1))
            self.stage_basis = nn.Parameter(torch.randn(D) * 0.01)

    def view(self, xhat, stage):
        if self.mode == "qinco_mirror_givens":
            return givens_view(xhat, self.stage_angle[stage])
        if self.mode == "qinco_additive_rank1":
            return xhat + self.stage_scale[stage] * self.stage_basis
        return xhat

    def encode(self, x):
        code0 = assign_to_codebook(x, self.codebook0.weight)
        xhat = self.codebook0.weight[code0]
        codes = [code0]
        for stage in range(M - 1):
            context = self.view(xhat, stage)
            q = self.step.codebook.weight
            qrep = q[None].expand(len(x), -1, -1)
            crep = context[:, None, :].expand(-1, K, -1)
            z = qrep.reshape(-1, D)
            cc = torch.cat((z, crep.reshape(-1, D)), dim=1)
            z = z + self.step.MLPconcat(cc)
            for block in self.step.residual_blocks:
                z = z + block(z)
            candidates = xhat[:, None, :] + z.reshape(len(x), K, D)
            idx = (x[:, None, :] - candidates).square().sum(-1).argmin(1)
            xhat = candidates[torch.arange(len(x)), idx]
            codes.append(idx)
        return torch.stack(codes, dim=1), xhat

    def decode(self, codes):
        xhat = self.codebook0(codes[:, 0])
        for stage in range(M - 1):
            xhat = xhat + self.step.decode(self.view(xhat, stage), codes[:, stage + 1])
        return xhat

    def forward(self, x):
        with torch.no_grad():
            codes, _ = self.encode(x)
        xhat = self.codebook0(codes[:, 0])
        losses = [((xhat - x) ** 2).sum()]
        for stage in range(M - 1):
            xhat = xhat + self.step.decode(self.view(xhat, stage), codes[:, stage + 1])
            losses.append(((xhat - x) ** 2).sum())
        return codes, xhat, torch.stack(losses)


class StaticRQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.codebooks = nn.ParameterList([nn.Parameter(torch.zeros(K, D)) for _ in range(M)])
    def encode(self, x):
        residual, codes = x, []
        for cb in self.codebooks:
            code = assign_to_codebook(residual, cb)
            codes.append(code)
            residual = residual - cb[code]
        codes = torch.stack(codes, dim=1)
        return codes, self.decode(codes)
    def decode(self, codes):
        return sum(cb[codes[:, i]] for i, cb in enumerate(self.codebooks))


def build_models(codebooks, seed):
    torch.manual_seed(seed + 1)
    native = QINCo(D, K, L, M, H)
    with torch.no_grad():
        native.codebook0.weight.copy_(codebooks[0])
        for i, step in enumerate(native.steps):
            step.codebook.weight.copy_(codebooks[i + 1])
    shared_init = torch.stack(codebooks[1:]).mean(0)
    models = {"qinco_native": native}
    for method in ["qinco_tied", "qinco_mirror_givens", "qinco_additive_rank1"]:
        torch.manual_seed(seed + 1)
        model = SharedStageQINCo(method)
        with torch.no_grad():
            model.codebook0.weight.copy_(codebooks[0])
            model.step.codebook.weight.copy_(shared_init)
        models[method] = model
    rq = StaticRQ()
    with torch.no_grad():
        for cb, init in zip(rq.codebooks, codebooks): cb.copy_(init)
    models["rq_static"] = rq
    return models


def train_model(model, x, seed):
    if isinstance(model, StaticRQ):
        return 0.0, {"updates": 0, "examples": 0, "loss_first_last": []}
    rng = np.random.default_rng(seed + 9000)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    model.train(); t0 = time.perf_counter(); losses = []
    xt = torch.from_numpy(x)
    for _ in range(UPDATES):
        idx = rng.choice(len(xt), BATCH, replace=False)
        batch = xt[torch.from_numpy(idx)]
        _, _, stage_losses = model(batch)
        loss = stage_losses.sum() / (len(batch) * D)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
    return time.perf_counter() - t0, {"updates": UPDATES, "examples": UPDATES * BATCH,
                                      "loss_first_last": [losses[0], losses[-1]]}


def pack_codes(codes):
    a = codes.astype(np.uint8)
    out = np.empty((len(a), M // 2), dtype=np.uint8)
    for i in range(M // 2): out[:, i] = a[:, 2*i] | (a[:, 2*i+1] << 4)
    return out.tobytes(order="C")


def unpack_codes(raw, n):
    a = np.frombuffer(raw, dtype=np.uint8).reshape(n, M // 2)
    out = np.empty((n, M), dtype=np.int64)
    for i in range(M // 2):
        out[:, 2*i] = a[:, i] & 15; out[:, 2*i+1] = (a[:, i] >> 4) & 15
    return out


def serialize_model(model, method, world_seed):
    arrays = {k: v.detach().cpu().numpy().astype("<f4") for k, v in model.state_dict().items()}
    meta = {"schema":"MA1035-model-v1","method":method,"d":D,"K":K,"M":M,"L":L,"hidden":H,
            "world_seed":world_seed,"upstream_repo_commit":"5a324954d5c9b3700d4407d6cc24c3db6e52890e",
            "upstream_model_source_sha256":"042a741b2574c32dc46e40e1ef737daaa961732b68b54e5060899b5a575a7d5a",
            "coordinate":"pairwise Givens on decoded prefix" if method=="qinco_mirror_givens" else method,
            "tensor_manifest":[{"name":k,"shape":list(v.shape),"dtype":"float32-le"} for k,v in arrays.items()]}
    return json.dumps(meta,sort_keys=True,separators=(",",":")).encode()+b"\n"+b"".join(v.tobytes(order="C") for v in arrays.values())


def exact_topk(q, db, k=10):
    dist = (q*q).sum(1)[:,None] + (db*db).sum(1)[None,:] - 2*q@db.T
    part = np.argpartition(dist, k-1, axis=1)[:, :k]
    order = np.take_along_axis(dist, part, axis=1).argsort(1)
    return np.take_along_axis(part, order, axis=1)


def metrics(xdb, xq, recon, truth, train_wall, encode_wall, decode_wall, search_wall, payload_bytes, model_bytes, index_bytes):
    r = np.mean((xdb-recon)**2) / (np.var(xdb)+1e-12)
    pred = exact_topk(xq, recon, 10)
    r1 = float(np.mean(pred[:,0] == truth[:,0]))
    r10 = float(np.mean([len(set(p)&set(t))/10 for p,t in zip(pred,truth)]))
    return {"normalized_mse":float(r),"recall_at_1":r1,"recall_at_10":r10,
            "training_wall_s":train_wall,"database_encode_wall_s":encode_wall,"database_decode_load_wall_s":decode_wall,
            "search_wall_s":search_wall,"query_qps":len(xq)/max(search_wall,1e-12),
            "resident_reconstruction_bytes":int(recon.nbytes),"serialized_model_bytes":model_bytes,
            "serialized_index_bytes":index_bytes,"total_serialized_bytes":payload_bytes,
            "full_float32_database_bytes":int(xdb.nbytes)}


def run(lane, seed):
    RUNS.mkdir(exist_ok=True)
    world = make_world(seed); xtrain = world["train"]; xdb=world["database"]; xq=world["queries"]
    codebooks = initialize_centroids(xtrain, seed+1)
    models=build_models(codebooks,seed); train_wall={}; train_info={}
    for method in METHODS: train_wall[method],train_info[method]=train_model(models[method],xtrain,seed)
    truth=exact_topk(xq,xdb,10); rows={}; payload_hash={}; code_hash={}
    for method in METHODS:
        model=models[method];model.eval()
        t0=time.perf_counter()
        with torch.no_grad(): codes,_=model.encode(torch.from_numpy(xdb))
        encode_wall=time.perf_counter()-t0;codes_np=codes.cpu().numpy().astype(np.uint8)
        model_raw=serialize_model(model,method,seed)
        code_raw=pack_codes(codes_np)
        index_meta=json.dumps({"schema":"MA1035-index-v1","method":method,"rows":len(xdb),"d":D,"M":M,"K":K,"codes_per_row":M,"packed_bits_per_code":4,"implicit_row_ids":True},sort_keys=True,separators=(",",":")).encode()+b"\n"
        index_raw=index_meta+code_raw
        model_path=RUNS/f"model_{seed}_{method}.bin"; index_path=RUNS/f"index_{seed}_{method}.bin"
        model_path.write_bytes(model_raw);index_path.write_bytes(index_raw)
        payload_hash[method]=hashlib.sha256(model_raw).hexdigest();code_hash[method]=hashlib.sha256(index_raw).hexdigest()
        t0=time.perf_counter()
        with torch.no_grad():recon=model.decode(torch.from_numpy(codes_np.astype(np.int64))).cpu().numpy().astype(np.float32)
        decode_wall=time.perf_counter()-t0
        t0=time.perf_counter();pred=exact_topk(xq,recon,10);search_wall=time.perf_counter()-t0
        # compute metrics with this already-measured prediction to avoid an unreported second search.
        nmse=float(np.mean((xdb-recon)**2)/(np.var(xdb)+1e-12))
        r1=float(np.mean(pred[:,0]==truth[:,0]));r10=float(np.mean([len(set(p)&set(t))/10 for p,t in zip(pred,truth)]))
        model_bytes=len(model_raw);index_bytes=len(index_raw);total=model_bytes+index_bytes
        rows[method]={"normalized_mse":nmse,"recall_at_1":r1,"recall_at_10":r10,
            "training_wall_s":train_wall[method],"training":train_info[method],"database_encode_wall_s":encode_wall,
            "database_decode_load_wall_s":decode_wall,"search_wall_s":search_wall,"query_qps":len(xq)/max(search_wall,1e-12),
            "resident_reconstruction_bytes":int(recon.nbytes),"serialized_model_bytes":model_bytes,"serialized_index_bytes":index_bytes,
            "total_serialized_bytes":total,"full_float32_database_bytes":int(xdb.nbytes),"code_count":int(codes_np.size),
            "model_payload_sha256":payload_hash[method],"index_payload_sha256":code_hash[method]}
    doc={"schema":"MA1035-v1","lane":lane,"seed":seed,"world":world["generation"],
         "shape":{"train":list(xtrain.shape),"database":list(xdb.shape),"queries":list(xq.shape)},
         "model_config":{"d":D,"K":K,"M":M,"L":L,"hidden":H,"updates":UPDATES,"batch":BATCH,"lr":LR},
         "source_sha256":hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
         "qinco_reference_sha256":hashlib.sha256(pathlib.Path(__file__).with_name('qinco_reference.py').read_bytes()).hexdigest(),
         "quantization_helper_sha256":hashlib.sha256(pathlib.Path(__file__).with_name('quantization.py').read_bytes()).hexdigest(),
         "results":rows,"totals":{"raw_database_bytes":int(xdb.nbytes),"exact_ground_truth":"brute-force Euclidean top-10 on uncompressed database"},
         "boundaries":["synthetic 32-D clustered-vector worlds only","QINCo-v1-structure model with small K/M and torch k-means init; not published checkpoint or paper reproduction","database scan uses generated reconstructions resident in RAM; serialized payload charges model plus packed indices and metadata"]}
    (RUNS/f"{lane}_{seed}.json").write_text(json.dumps(doc,indent=2)+'\n')
    print(json.dumps(doc,indent=2))


if __name__ == "__main__":
    ap=argparse.ArgumentParser();ap.add_argument('--lane',choices=['development','audit'],required=True);ap.add_argument('--seed',type=int,required=True);a=ap.parse_args();run(a.lane,a.seed)
