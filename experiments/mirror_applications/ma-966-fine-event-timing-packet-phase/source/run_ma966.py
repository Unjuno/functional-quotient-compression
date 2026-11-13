"""CPU screen for MA-966 on the public D1 Vernier trials.

Downloads only the two required members from the official Zenodo ZIP with HTTP
Range requests; the 715 MB dataset archive is never downloaded in full.
"""
import csv, hashlib, io, json, math, pickle, random, struct, time, urllib.request, zlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/tmp/ma966_data')
DATA.mkdir(parents=True, exist_ok=True)
RECORD = 20973757
ZIP_URL = f'https://zenodo.org/api/records/{RECORD}/files/datasets.zip/content'
TOTAL = 715632615
CD_OFFSET, CD_SIZE = 715621894, 10721
TARGETS = {'datasets/ebtMNIST-D1/Vernier/events.pkl',
           'datasets/ebtMNIST-D1/Vernier/labels.pkl'}
WINDOWS = (80.0, 100.0, 120.0, 140.0)
NEVENTS = 48
PERIOD_MS = 128.0
TRAIN_LABELS = {13, 14, 26, 27}
DEV_LABELS = {15, 25}
FRESH_LABELS = {16, 17, 18, 19, 21, 22, 23, 24}
SEEDS = (96611, 96613, 96617)


def get_range(start, end):
    req = urllib.request.Request(ZIP_URL, headers={'Range': f'bytes={start}-{end}', 'User-Agent': 'MA966-worker/1.0'})
    with urllib.request.urlopen(req, timeout=60) as r:
        if r.status != 206:
            raise RuntimeError(f'expected HTTP 206 for range, got {r.status}')
        return r.read()


def fetch_members():
    if all((DATA / Path(n).name).exists() for n in TARGETS):
        return
    cd = get_range(CD_OFFSET, CD_OFFSET + CD_SIZE - 1)
    entries = {}
    pos = 0
    while pos + 46 <= len(cd):
        if cd[pos:pos+4] != b'PK\x01\x02':
            pos += 1
            continue
        f = struct.unpack_from('<4s6H3I5H2I', cd, pos)
        method, crc, csize, usize = f[4], f[7], f[8], f[9]
        fnl, exl, coml, local = f[10], f[11], f[12], f[16]
        name = cd[pos+46:pos+46+fnl].decode()
        entries[name] = (method, crc, csize, usize, local, fnl, exl)
        pos += 46 + fnl + exl + coml
    for name in sorted(TARGETS):
        method, crc, csize, usize, local, _, _ = entries[name]
        local_head = get_range(local, local + 29)
        if local_head[:4] != b'PK\x03\x04':
            raise RuntimeError(f'bad local ZIP header for {name}')
        fnl, exl = struct.unpack_from('<HH', local_head, 26)
        start = local + 30 + fnl + exl
        comp = get_range(start, start + csize - 1)
        if method == 8:
            raw = zlib.decompress(comp, -15)
        elif method == 0:
            raw = comp
        else:
            raise RuntimeError(f'unsupported zip method {method}')
        if len(raw) != usize or (zlib.crc32(raw) & 0xffffffff) != crc:
            raise RuntimeError(f'ZIP size/CRC mismatch for {name}')
        (DATA / Path(name).name).write_bytes(raw)


def load_data():
    fetch_members()
    ep = DATA / 'events.pkl'
    lp = DATA / 'labels.pkl'
    event_bytes, label_bytes = ep.read_bytes(), lp.read_bytes()
    events, labels = pickle.loads(event_bytes), np.asarray(pickle.loads(label_bytes), dtype=np.int64).reshape(-1)
    if len(events) != len(labels):
        raise RuntimeError(f'events/labels length mismatch {len(events)} != {len(labels)}')
    x = np.zeros((len(events), len(WINDOWS), NEVENTS, 4), dtype=np.float32)
    valid_counts = np.zeros((len(events), len(WINDOWS)), dtype=np.int32)
    # Mirror the official loader's 1e-3 timestamp scaling and 48-event start-time windows.
    for i, trial in enumerate(events):
        t = np.asarray(trial['timestamp'], dtype=np.float64) * 1e-3
        xx = np.asarray(trial['x'], dtype=np.float32)
        yy = np.asarray(trial['y'], dtype=np.float32)
        pp = np.asarray(trial['polarity'], dtype=np.float32)
        order = np.argsort(t, kind='stable')
        t, xx, yy, pp = t[order], xx[order], yy[order], pp[order]
        for wi, start_ms in enumerate(WINDOWS):
            start = int(np.searchsorted(t, start_ms, side='left'))
            start = min(start, max(0, len(t) - NEVENTS))
            sl = slice(start, min(start + NEVENTS, len(t)))
            n = len(t[sl]); valid_counts[i, wi] = n
            if n:
                x[i, wi, :n] = np.column_stack([t[sl], xx[sl], yy[sl], pp[sl]])
    meta = {
        'dataset_record': RECORD,
        'events_sha256': hashlib.sha256(event_bytes).hexdigest(),
        'labels_sha256': hashlib.sha256(label_bytes).hexdigest(),
        'event_trials': len(events), 'labels': {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
        'min_valid_events': int(valid_counts.min()),
        'timestamp_unit_after_official_rescale': 'ms',
        'windows_ms': list(WINDOWS), 'events_per_window': NEVENTS
    }
    return x, labels, meta


class EventEncoder(nn.Module):
    def __init__(self, mode='raw'):
        super().__init__()
        self.mode = mode
        self.input = nn.Linear((3 if mode == 'phaseonly' else 4) + (2 if mode in ('periodic', 'mirror', 'offset', 'phaseonly') else 0), 32)
        layer = nn.TransformerEncoderLayer(32, 4, 64, dropout=0.0, batch_first=True, norm_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(32)
        self.head = nn.Linear(32, 2)
        if mode in ('mirror', 'offset'):
            # Both are in radians. Offset control uses the same exact coordinate as m.
            key = 'm_window' if mode == 'mirror' else 'native_offset_phase'
            self.register_parameter(key, nn.Parameter(torch.zeros(4)))

    def forward(self, x, window_ids):
        # x: [batch, event, (t_ms,x,y,polarity)]; preserve raw time except phaseonly.
        t, xx, yy, pol = x.unbind(-1)
        base = torch.stack([(t - 110.0) * 0.1, xx - 35.0, yy - 26.0, pol], dim=-1)
        if self.mode in ('periodic', 'mirror', 'offset', 'phaseonly'):
            angle = (2 * math.pi / PERIOD_MS) * t
            if self.mode in ('mirror', 'offset'):
                coord = self.m_window if self.mode == 'mirror' else self.native_offset_phase
                angle = angle + coord[window_ids][:, None]
            periodic = torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)
            if self.mode == 'phaseonly':
                base = base[..., 1:]
            feat = torch.cat([base, periodic], dim=-1)
        else:
            feat = base
        h = self.input(feat)
        h = self.encoder(h)
        return self.head(self.norm(h).mean(dim=1))


class Independent(nn.Module):
    def __init__(self):
        super().__init__()
        self.models = nn.ModuleList([EventEncoder('raw') for _ in WINDOWS])

    def forward(self, x, window_ids):
        out = torch.zeros((len(x), 2), device=x.device)
        for wi, model in enumerate(self.models):
            mask = window_ids == wi
            if mask.any():
                out[mask] = model(x[mask], window_ids[mask])
        return out


def select_indices(labels, allowed):
    return np.flatnonzero(np.isin(labels, list(allowed)))


def minibatches(indices, batch_size, rng):
    order = np.asarray(indices)[rng.permutation(len(indices))]
    for start in range(0, len(order), batch_size):
        yield order[start:start+batch_size]


def flatten_trials(x, labels, idx):
    rows = []
    for i in idx:
        for w in range(len(WINDOWS)):
            rows.append((x[i, w], w, int(labels[i] > 20), int(labels[i])))
    return rows


def tensor_batch(rows, device='cpu'):
    xx = torch.from_numpy(np.stack([r[0] for r in rows])).to(device)
    ww = torch.tensor([r[1] for r in rows], dtype=torch.long, device=device)
    yy = torch.tensor([r[2] for r in rows], dtype=torch.long, device=device)
    labs = np.asarray([r[3] for r in rows], dtype=np.int64)
    return xx, ww, yy, labs


def accuracy_by_offset(pred, labels):
    out = {}
    for off in range(1, 8):
        keep = np.abs(labels - 20) == off
        if keep.any():
            out[str(off)] = float(np.mean(pred[keep] == (labels[keep] > 20)))
    return out


def payload(model, config):
    buf = io.BytesIO()
    torch.save({'state_dict': model.state_dict(), 'config': config}, buf)
    return buf.getvalue()


def evaluate(model, x, labels, idx, jitter_sigma=0.0, seed=0):
    rows = flatten_trials(x, labels, idx)
    xx, ww, yy, labs = tensor_batch(rows)
    xeval = xx.clone()
    if jitter_sigma:
        g = torch.Generator().manual_seed(seed)
        xeval[..., 0] += torch.randn(xeval[..., 0].shape, generator=g) * jitter_sigma
    model.eval()
    with torch.no_grad():
        logits = model(xeval, ww)
    pred = logits.argmax(-1).cpu().numpy().astype(np.int64)
    by = accuracy_by_offset(pred, labs)
    vals = list(by.values())
    return {'accuracy_by_offset': by, 'macro_accuracy': float(np.mean(vals)) if vals else None,
            'predictions': pred, 'labels': labs}


def fit_one(method, x, labels, tr_idx, dev_idx, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    torch.set_num_threads(1)
    model = Independent() if method == 'independent' else EventEncoder(method)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    tr_rows = flatten_trials(x, labels, tr_idx)
    dev_rows = flatten_trials(x, labels, dev_idx)
    dev_x, dev_w, dev_y, dev_labs = tensor_batch(dev_rows)
    best_acc, best_epoch, best_state, stale = -1.0, 0, None, 0
    updates = 0; started = time.perf_counter()
    for epoch in range(30):
        model.train(); rng = np.random.default_rng(seed + epoch)
        for indices in minibatches(np.arange(len(tr_rows)), 64, rng):
            batch = [tr_rows[i] for i in indices]
            bx, bw, by, _ = tensor_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            logits = model(bx, bw)
            loss = nn.functional.cross_entropy(logits, by)
            loss.backward(); optimizer.step(); updates += 1
        model.eval()
        with torch.no_grad():
            acc = float((model(dev_x, dev_w).argmax(-1) == dev_y).float().mean())
        if acc > best_acc:
            best_acc, best_epoch, stale = acc, epoch + 1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 6: break
    model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    cfg = {'method': method, 'seed': seed, 'period_ms': PERIOD_MS, 'windows_ms': WINDOWS,
           'events_per_window': NEVENTS, 'architecture': 'Transformer d32/layer1/head4/ff64'}
    blob = payload(model, cfg)
    reloaded = EventEncoder(method) if method != 'independent' else Independent()
    ckpt = torch.load(io.BytesIO(blob), map_location='cpu', weights_only=False)
    reloaded.load_state_dict(ckpt['state_dict']); reloaded.eval()
    dev_replay = evaluate(reloaded, x, labels, dev_idx)
    train_tok = int(len(tr_rows) * NEVENTS * best_epoch)
    return reloaded, {'dev_accuracy': best_acc, 'best_epoch': best_epoch, 'optimizer_updates': updates,
                      'encoder_token_presentations': train_tok, 'wall_time_s': elapsed,
                      'serialized_bytes': len(blob), 'payload_sha256': hashlib.sha256(blob).hexdigest(),
                      'serialization_replay_accuracy': dev_replay['macro_accuracy']}


def ridge_baseline(x, labels, tr_idx, eval_idx):
    rows = flatten_trials(x, labels, tr_idx)
    a = np.stack([r[0].reshape(-1) for r in rows]).astype(np.float64)
    y = np.asarray([r[2] * 2 - 1 for r in rows], dtype=np.float64)
    # Ridge(alpha=1), closed form with centered feature matrix and intercept.
    mu = a.mean(0); a0 = a - mu
    coef = np.linalg.solve(a0.T @ a0 + np.eye(a0.shape[1]), a0.T @ y)
    bias = y.mean() - mu @ coef
    eval_rows = flatten_trials(x, labels, eval_idx)
    ex = np.stack([r[0].reshape(-1) for r in eval_rows]).astype(np.float64)
    pred = (ex @ coef + bias) > 0
    labs = np.asarray([r[3] for r in eval_rows])
    return accuracy_by_offset(pred, labs), {'ridge_payload_bytes': len(coef.tobytes()) + len(np.asarray([bias], dtype=np.float64).tobytes()),
                                            'ridge_training_rows': len(rows), 'ridge_eval_rows': len(eval_rows)}


def main():
    x, labels, meta = load_data()
    train_idx = select_indices(labels, TRAIN_LABELS)
    dev_idx = select_indices(labels, DEV_LABELS)
    audit_idx = select_indices(labels, FRESH_LABELS)
    # Do not train or score on any other labels; verify split isolation now.
    assert set(labels[train_idx]) == TRAIN_LABELS
    assert set(labels[dev_idx]) == DEV_LABELS
    assert set(labels[audit_idx]) == FRESH_LABELS
    assert len(set(train_idx) & set(dev_idx)) == 0 and len(set(train_idx) & set(audit_idx)) == 0
    assert len(set(dev_idx) & set(audit_idx)) == 0
    meta['split_counts'] = {'train_trials': int(len(train_idx)), 'development_trials': int(len(dev_idx)), 'fresh_trials': int(len(audit_idx))}
    (ROOT / 'source' / 'DATA_PROVENANCE.json').write_text(json.dumps(meta, indent=2) + '\n')
    methods = ['raw', 'periodic', 'mirror', 'offset', 'phaseonly', 'independent']
    results = []
    models = {}
    # Development is the only split opened for checkpoint selection.
    for method in methods:
        model, m = fit_one(method, x, labels, train_idx, dev_idx, 96601)
        models[method] = model
        results.append({'method': method, 'seed': 96601, **m, 'split': 'development'})
        print('DEV', method, json.dumps(results[-1], sort_keys=True), flush=True)
    # Check the m and native-offset implementations are exact function aliases.
    test_x = torch.from_numpy(x[audit_idx[:16], 0]); wid = torch.zeros(len(test_x), dtype=torch.long)
    a = models['mirror']; b = models['offset']
    with torch.no_grad():
        # Align semantic parameter names and verify the same code path is numerically identical.
        b.native_offset_phase.copy_(a.m_window)
        alias_err = float((a(test_x, wid) - b(test_x, wid)).abs().max())
    if alias_err > 1e-7:
        raise AssertionError(f'phase/time-offset alias error {alias_err}')
    (ROOT / 'source' / 'ALIAS_AUDIT.json').write_text(json.dumps({'max_abs_logit_error': alias_err, 'tolerance': 1e-7, 'exact_math': 'sin/cos rotation by m equals adding native time offset m*P/(2π)'}, indent=2) + '\n')
    # Development is complete. Fresh labels are first evaluated below; never tune afterward.
    ledger = []
    core_path = ROOT / 'RESULTS_CORE.csv'
    dev_selection = {r['method']: r['best_epoch'] for r in results}
    for method in methods:
        for seed in SEEDS:
            model, m = fit_one(method, x, labels, train_idx, dev_idx, seed)
            # Locked fresh evaluation. The development labels are used only for early stopping.
            ev = evaluate(model, x, labels, audit_idx)
            row = {'condition': method, 'seed': seed, 'train_trials': len(train_idx), 'development_trials': len(dev_idx),
                   'audit_trials': len(audit_idx), 'serialized_bytes': m['serialized_bytes'],
                   'optimizer_updates': m['optimizer_updates'], 'encoder_token_presentations': m['encoder_token_presentations'],
                   'wall_time_s': m['wall_time_s'], 'primary_metric': 'fresh macro sign accuracy offsets 1..4',
                   'primary_value': ev['macro_accuracy'], 'secondary_metric': 'serialized replay; offset accuracies',
                   'secondary_value': json.dumps({'replay': m['serialization_replay_accuracy'], 'by_offset': ev['accuracy_by_offset']}, separators=(',', ':')),
                   'status_note': 'three seeds on one acquisition; not independent worlds'}
            ledger.append({'condition': method, 'seed': seed, **m, 'fresh_macro_accuracy': ev['macro_accuracy'],
                           'fresh_accuracy_by_offset': ev['accuracy_by_offset']})
            with core_path.open('a', newline='') as f:
                csv.DictWriter(f, fieldnames=next(csv.reader([core_path.read_text().splitlines()[0]]))).writerow(row)
            print('FRESH', method, seed, json.dumps(ledger[-1], sort_keys=True), flush=True)
            if method == 'raw':
                jitter_rows = {}
                for sigma in (0.25, 0.5, 1.0, 2.0):
                    je = evaluate(model, x, labels, audit_idx, sigma, seed + int(sigma*100))
                    jitter_rows[str(sigma)] = {'macro_accuracy': je['macro_accuracy'], 'by_offset': je['accuracy_by_offset']}
                (ROOT / 'source' / f'JITTER_{seed}.json').write_text(json.dumps(jitter_rows, indent=2) + '\n')
    (ROOT / 'source' / 'FRESH_RESULTS.json').write_text(json.dumps(ledger, indent=2) + '\n')
    # Native raw-event ridge diagnostic on the same time windows; no pretrained features.
    by, rm = ridge_baseline(x, labels, train_idx, audit_idx)
    (ROOT / 'source' / 'RAW_RIDGE.json').write_text(json.dumps({'fresh_accuracy_by_offset': by, **rm}, indent=2) + '\n')
    print('RIDGE', json.dumps({'accuracy_by_offset': by, **rm}, sort_keys=True), flush=True)
    print('DATA', json.dumps(meta, sort_keys=True), flush=True)

if __name__ == '__main__':
    main()
