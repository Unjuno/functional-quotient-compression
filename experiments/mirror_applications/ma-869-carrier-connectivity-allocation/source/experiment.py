"""Synthetic carrier-connectivity allocation screen for MA-869."""
from __future__ import annotations

import json
import math
import struct
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

N_AGENTS = 8
N_DIMS = 4
MESSAGE = struct.Struct("<BBf")


@dataclass
class World:
    seed: int
    cut: bool
    near_is_low_noise: list[bool]
    reachable: list[bool]
    sigmas: list[float]


def make_world(seed: int) -> World:
    rng = np.random.default_rng(seed)
    cut = bool(rng.random() < 0.5)
    near_is_low = [bool(x) for x in rng.integers(0, 2, size=N_DIMS)]
    reachable = [True] * 4 + [not cut] * 4
    sigmas: list[float] = []
    for agent in range(N_AGENTS):
        dim, far = agent % 4, agent >= 4
        low_near = near_is_low[dim]
        sigmas.append(0.25 if (not far) == low_near else 0.75)
    return World(seed, cut, near_is_low, reachable, sigmas)


def policy_agents(world: World, policy: str) -> list[int | None]:
    result: list[int | None] = []
    for dim in range(N_DIMS):
        pair = [dim, dim + 4]
        if policy == "central":
            result.append(None)
        elif policy == "best_ignoring_connectivity":
            result.append(min(pair, key=lambda a: world.sigmas[a]))
        elif policy == "connectivity_aware":
            candidates = [a for a in pair if world.reachable[a]]
            result.append(min(candidates, key=lambda a: world.sigmas[a]) if candidates else None)
        elif policy == "random_reachable":
            candidates = [a for a in pair if world.reachable[a]]
            result.append(candidates[(world.seed + dim) % len(candidates)] if candidates else None)
        else:
            raise ValueError(policy)
    return result


def _estimate(z: np.ndarray, observations: np.ndarray, agents: list[int], sigmas: list[float]) -> tuple[np.ndarray, int]:
    precision = np.ones(N_DIMS, dtype=np.float64)
    weighted = np.zeros_like(z)
    for agent in agents:
        dim = agent % 4
        variance = sigmas[agent] ** 2
        precision[dim] += 1.0 / variance
        weighted[:, dim] += observations[:, agent] / variance
    estimate = weighted / precision[None, :]
    return estimate, len(agents)


def evaluate(seed: int, draws: int = 4096) -> dict[str, Any]:
    start = time.perf_counter()
    world = make_world(seed)
    rng = np.random.default_rng(seed + 100_003)
    z = rng.normal(size=(draws, N_DIMS))
    noise = rng.normal(size=(draws, N_AGENTS)) * np.asarray(world.sigmas)[None, :]
    observations = np.empty((draws, N_AGENTS), dtype=np.float64)
    for agent in range(N_AGENTS):
        observations[:, agent] = z[:, agent % 4] + noise[:, agent]

    selected = policy_agents(world, "connectivity_aware")
    selected_ignore = policy_agents(world, "best_ignoring_connectivity")
    selected_random = policy_agents(world, "random_reachable")
    results: dict[str, Any] = {}

    def measure(name: str, selection: list[int | None], collect_all: bool = False) -> None:
        delivered: list[int] = []
        violations = 0
        for item in selection:
            if collect_all:
                continue
            if item is None:
                continue
            if not world.reachable[item]:
                violations += 1
            else:
                delivered.append(item)
        if collect_all:
            delivered = [a for a in range(N_AGENTS) if world.reachable[a]]
        estimate, _ = _estimate(z, observations, delivered, world.sigmas)
        nmse = float(np.mean((estimate - z) ** 2))
        hop_count = sum(a + 1 for a in delivered)
        messages = len(delivered)
        results[name] = {
            "nmse": nmse,
            "messages": messages,
            "message_hops": hop_count,
            "message_hop_bytes": hop_count * MESSAGE.size,
            "hop_latency": max((a + 1 for a in delivered), default=0),
            "disconnected_selection_violations": violations,
            "posterior_scalar_ops": 6 * len(delivered) + 4,
        }

    measure("central", [], collect_all=True)
    measure("best_ignoring_connectivity", selected_ignore)
    measure("connectivity_aware", selected)
    measure("random_reachable", selected_random)

    native, mirror = serialize_policy(world, "native"), serialize_policy(world, "mirror")
    results["policy_storage"] = {
        "schema_bytes": len(serialize_schema(world)),
        "native_total_bytes": len(native),
        "mirror_total_bytes": len(mirror),
        "native_policy_bytes": len(native) - len(serialize_schema(world)),
        "mirror_policy_bytes": len(mirror) - len(serialize_schema(world)),
        "native_mirror_behavior_identical": [a if a is not None else 255 for a in policy_agents(world, "connectivity_aware")] == list(serialize_policy(world, "mirror")[-N_DIMS:]),
        "message_bytes": MESSAGE.size,
    }
    results["world"] = {"seed": seed, "cut": world.cut, "near_is_low_noise": world.near_is_low_noise, "sigmas": world.sigmas}
    results["wall_seconds"] = time.perf_counter() - start
    return results


def serialize_schema(world: World) -> bytes:
    # 1 format, 1 cut, 4 near/far low-noise flags, 8 reliability class IDs.
    reliability = bytes(0 if sigma == 0.25 else 1 for sigma in world.sigmas)
    return struct.pack("<BB", 1, int(world.cut)) + bytes(world.near_is_low_noise) + reliability


def serialize_policy(world: World, kind: str) -> bytes:
    schema = serialize_schema(world)
    if kind == "native":
        # Fixed algorithm ID; caller supplies paid schema above.
        return schema + b"N" + bytes([1, 0])
    if kind == "mirror":
        # Four task addresses; the address table is fully charged.
        addresses = [a if a is not None else 255 for a in policy_agents(world, "connectivity_aware")]
        return schema + b"M" + bytes(addresses)
    raise ValueError(kind)


def run(seeds: list[int], draws: int = 4096) -> dict[str, Any]:
    rows = [evaluate(seed, draws) for seed in seeds]
    summary: dict[str, Any] = {"experiment_id": "MA-869", "draws_per_seed": draws, "seeds": seeds, "rows": rows}
    for policy in ("central", "best_ignoring_connectivity", "connectivity_aware", "random_reachable"):
        vals = [row[policy] for row in rows]
        summary[policy] = {key: float(np.mean([v[key] for v in vals])) for key in ("nmse", "messages", "message_hops", "message_hop_bytes", "hop_latency", "posterior_scalar_ops")}
        summary[policy]["max_disconnected_selection_violations"] = max(v["disconnected_selection_violations"] for v in vals)
    summary["storage"] = {key: int(np.mean([r["policy_storage"][key] for r in rows])) for key in ("schema_bytes", "native_total_bytes", "mirror_total_bytes", "native_policy_bytes", "mirror_policy_bytes", "message_bytes")}
    summary["max_wall_seconds"] = max(r["wall_seconds"] for r in rows)
    summary["provenance"] = {"source": "source/experiment.py", "message_format": MESSAGE.format, "protocol_seeds": seeds}
    return summary


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
