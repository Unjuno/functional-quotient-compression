"""Bounded CPU feasibility profile on pinned MOMENT-1-base; no metric outputs."""
from __future__ import annotations

import argparse
import json
import time

import torch
from torch import nn


class GatedLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 2):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=5**0.5)
        nn.init.normal_(self.b.weight, mean=0.0, std=0.01)
        self.gate = nn.Parameter(torch.ones(()))

    @property
    def weight(self):
        # Transformers' T5 gated-FFN dispatch checks ``wo.weight``.
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        return self.base(x) + self.gate * self.b(self.a(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(4)
    torch.manual_seed(105201)
    from momentfm import MOMENTPipeline

    started = time.perf_counter()
    model = MOMENTPipeline.from_pretrained(
        args.checkpoint,
        model_kwargs={
            "task_name": "forecasting",
            "forecast_horizon": 192,
            "freeze_encoder": True,
            "freeze_embedder": True,
            "freeze_head": False,
        },
        local_files_only=True,
    )
    model.init()
    if hasattr(model.encoder, "gradient_checkpointing_disable"):
        model.encoder.gradient_checkpointing_disable()
    loaded_s = time.perf_counter() - started

    targets = ("q", "k", "v", "o", "wi_0", "wi_1", "wo")
    wrapped = []
    for name, module in list(model.encoder.named_modules()):
        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in targets:
            parent_name, _, leaf = name.rpartition(".")
            parent = model.encoder.get_submodule(parent_name) if parent_name else model.encoder
            parent._modules[leaf] = GatedLoRALinear(module)
            wrapped.append(name)
    assert len(wrapped) == 84, f"expected 84 TRACE LoRA sites, got {len(wrapped)}"

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=2e-4)
    x = torch.randn(args.batch, 1, 512)
    mask = torch.ones(args.batch, 512)
    times = []
    for i in range(3):
        optimizer.zero_grad(set_to_none=True)
        t = time.perf_counter()
        pred = model(x_enc=x, input_mask=mask).forecast
        pred.square().mean().backward()
        optimizer.step()
        times.append(time.perf_counter() - t)
    result = {
        "checkpoint": args.checkpoint,
        "batch": args.batch,
        "trace_lora_sites": len(wrapped),
        "trainable_parameters": sum(p.numel() for p in params),
        "load_seconds": loaded_s,
        "step_seconds": times,
        "steady_step_seconds": times[-1],
        "forecast_shape": list(pred.shape),
        "cpu_threads": torch.get_num_threads(),
        "synthetic_input_only": True,
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
