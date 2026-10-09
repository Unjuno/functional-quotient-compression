"""MA-160 fixed-weight quantized residual reconstruction screen."""
import io
import json
import math
import struct
import time

import torch

D, ROLES = 16, 4
ANGLES = torch.tensor([-.60, -.20, .20, .60], dtype=torch.float32)
COEFFS = torch.tensor([-.60, -.20, .20, .60], dtype=torch.float32)
METHODS = ("independent_int4", "mirror_no_residual", "mirror_shared_residual",
           "mirror_private_int4", "mirror_private_rank2", "independent_qer_rank2", "fp32_untied")


def rotation(theta):
    c, s = torch.cos(theta), torch.sin(theta)
    r = torch.eye(D)
    r[0, 0], r[0, 1], r[1, 0], r[1, 1] = c, -s, s, c
    return r


def quant4(w):
    scale = float(w.abs().max().clamp_min(1e-8) / 7)
    q = torch.round(w / scale).clamp(-7, 7).to(torch.int8)
    return q, scale


def _pack4(q):
    v = q.to(torch.int16).reshape(-1).clamp(-7, 7)
    n = torch.where(v < 0, v + 16, v).to(torch.uint8)
    if n.numel() % 2:
        n = torch.cat((n, torch.zeros(1, dtype=torch.uint8)))
    return (n[0::2] | (n[1::2] << 4)).numpy().tobytes()


def _unpack4(raw, count):
    b = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    v = torch.empty(b.numel() * 2, dtype=torch.int16)
    v[0::2], v[1::2] = (b & 15).to(torch.int16), (b >> 4).to(torch.int16)
    v = torch.where(v >= 8, v - 16, v)
    return v[:count].to(torch.int8)


def _tensor_bytes(t, dtype):
    if dtype == "i4":
        return _pack4(t), 0
    if dtype == "f16":
        return t.contiguous().half().numpy().tobytes(), 1
    return t.contiguous().float().numpy().tobytes(), 2


def serialize(method, entries):
    """Entry format charges names, shapes, scales, dtypes and all tensor data."""
    out = bytearray(b"MA160V1\0")
    mb = method.encode("ascii")
    out.extend(struct.pack("<BH", len(mb), len(entries)))
    out.extend(mb)
    for name, dtype, tensor, scale in entries:
        nb = name.encode("ascii")
        raw, type_id = _tensor_bytes(tensor, dtype)
        out.extend(struct.pack("<BB", len(nb), type_id))
        out.extend(nb)
        out.extend(struct.pack("<B", tensor.ndim))
        out.extend(struct.pack("<" + "H" * tensor.ndim, *tensor.shape))
        if dtype == "i4":
            out.extend(struct.pack("<f", scale))
        out.extend(struct.pack("<I", len(raw)))
        out.extend(raw)
    return bytes(out)


def deserialize(payload):
    view, off = memoryview(payload), 8
    method_len, count = struct.unpack_from("<BH", view, off); off += 3
    method = bytes(view[off:off + method_len]).decode("ascii"); off += method_len
    entries = {}
    for _ in range(count):
        name_len, type_id = struct.unpack_from("<BB", view, off); off += 2
        name = bytes(view[off:off + name_len]).decode("ascii"); off += name_len
        ndim = view[off]; off += 1
        shape = struct.unpack_from("<" + "H" * ndim, view, off); off += 2 * ndim
        scale = 1.0
        if type_id == 0:
            scale = struct.unpack_from("<f", view, off)[0]; off += 4
        raw_len = struct.unpack_from("<I", view, off)[0]; off += 4
        raw = bytes(view[off:off + raw_len]); off += raw_len
        if type_id == 0:
            tensor = _unpack4(raw, math.prod(shape)).reshape(shape).float() * scale
        else:
            dtype = torch.float16 if type_id == 1 else torch.float32
            tensor = torch.frombuffer(bytearray(raw), dtype=dtype).clone().reshape(shape).float()
        entries[name] = tensor
    if off != len(view):
        raise ValueError("trailing bytes in payload")
    return method, entries


def _rank2(a):
    u, s, vh = torch.linalg.svd(a, full_matrices=False)
    root = s[:2].clamp_min(0).sqrt()
    return (u[:, :2] * root).half(), (root[:, None] * vh[:2]).half()


def _add_q4(entries, name, tensor):
    q, scale = quant4(tensor)
    entries.append((name, "i4", q, scale))


def make_world(seed, aligned):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(D, D, generator=g) / math.sqrt(D)
    residual = torch.randn(D, D, generator=g) / (2 * math.sqrt(D))
    if aligned:
        targets = []
        for i, (a, c) in enumerate(zip(ANGLES, COEFFS)):
            r = rotation(a)
            noise = .012 * torch.randn(D, D, generator=g) / math.sqrt(D)
            targets.append(r @ (base + c * residual + noise) @ r.T)
        targets = torch.stack(targets)
    else:
        targets = torch.randn(ROLES, D, D, generator=g) / math.sqrt(D)
    xt = torch.randn(512, D, generator=g)
    return targets, base, residual, xt


def _estimate_shared(targets):
    unrot = torch.stack([rotation(a).T @ targets[i] @ rotation(a) for i, a in enumerate(ANGLES)])
    x = torch.stack((torch.ones(ROLES), COEFFS), dim=1)
    pinv = torch.linalg.pinv(x)
    return torch.einsum("kr,rij->kij", pinv, unrot)[0], torch.einsum("kr,rij->kij", pinv, unrot)[1]


def build(method, targets):
    entries = []
    if method == "fp32_untied":
        entries.append(("weights", "f32", targets, 1.0))
    elif method in ("independent_int4", "independent_qer_rank2"):
        for i, w in enumerate(targets):
            q, scale = quant4(w); qhat = q.float() * scale
            if method == "independent_int4":
                entries.append((f"w{i}", "i4", q, scale))
            else:
                left, right = _rank2(w - qhat)
                entries.extend([(f"w{i}", "i4", q, scale), (f"l{i}", "f16", left, 1.0), (f"r{i}", "f16", right, 1.0)])
    else:
        b, e = _estimate_shared(targets)
        _add_q4(entries, "base", b)
        entries.append(("angles", "f32", ANGLES, 1.0))
        if method == "mirror_shared_residual":
            _add_q4(entries, "residual", e)
            entries.append(("coeffs", "f32", COEFFS, 1.0))
        elif method == "mirror_no_residual":
            pass
        elif method == "mirror_private_int4":
            unrot = torch.stack([rotation(a).T @ targets[i] @ rotation(a) for i, a in enumerate(ANGLES)])
            for i in range(ROLES): _add_q4(entries, f"r{i}", unrot[i] - b)
        elif method == "mirror_private_rank2":
            unrot = torch.stack([rotation(a).T @ targets[i] @ rotation(a) for i, a in enumerate(ANGLES)])
            for i in range(ROLES):
                left, right = _rank2(unrot[i] - b)
                entries.extend([(f"l{i}", "f16", left, 1.0), (f"r{i}", "f16", right, 1.0)])
        else:
            raise ValueError(method)
    payload = serialize(method, entries)
    return decode(payload), payload


def decode(payload):
    method, e = deserialize(payload)
    if method == "fp32_untied": return e["weights"]
    if method == "independent_int4": return torch.stack([e[f"w{i}"] for i in range(ROLES)])
    if method == "independent_qer_rank2":
        return torch.stack([e[f"w{i}"] + e[f"l{i}"] @ e[f"r{i}"] for i in range(ROLES)])
    base, angles = e["base"], e["angles"]
    mats = []
    for i, a in enumerate(angles):
        w = base
        if method == "mirror_shared_residual": w = w + e["coeffs"][i] * e["residual"]
        elif method == "mirror_private_int4": w = w + e[f"r{i}"]
        elif method == "mirror_private_rank2": w = w + e[f"l{i}"] @ e[f"r{i}"]
        r = rotation(a); mats.append(r @ w @ r.T)
    return torch.stack(mats)


def run_one(method, seed, condition):
    targets, _, _, xt = make_world(seed, condition == "aligned")
    t0 = time.perf_counter(); decoded, payload = build(method, targets); encode_decode_s = time.perf_counter() - t0
    err = decoded - targets
    rel = float(err.norm() / targets.norm())
    activation_mse = float(torch.einsum("rij,nj->rni", err, xt).square().mean())
    # Reconstruction MAC proxy; training is post-training and has zero optimizer updates.
    if method == "independent_int4": macs = ROLES * D * D
    elif method == "mirror_no_residual": macs = D * D + ROLES * 2 * D ** 3
    elif method == "mirror_shared_residual": macs = 2 * D * D + ROLES * (2 * D ** 3 + D * D)
    elif method == "mirror_private_int4": macs = (ROLES + 1) * D * D + ROLES * 2 * D ** 3
    elif method == "mirror_private_rank2": macs = D * D + ROLES * (2 * D * D * 2 + 2 * D ** 3)
    elif method == "independent_qer_rank2": macs = ROLES * (D * D + 2 * D * D * 2)
    else: macs = ROLES * D * D
    return {"method": method, "seed": seed, "condition": condition, "relative_fro": rel,
            "activation_mse": activation_mse, "payload_bytes": len(payload), "decode_macs": macs,
            "encode_decode_wall_s": encode_decode_s, "examples": 512, "optimizer_updates": 0,
            "tokens": 0}


def run(seed, condition):
    return [run_one(method, seed, condition) for method in METHODS]
