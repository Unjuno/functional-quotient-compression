import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

from experiment import MESSAGE, evaluate, make_world, policy_agents, serialize_policy


def test_wire_format_and_policy_storage_are_actual_bytes():
    world = make_world(701)
    assert MESSAGE.size == 6
    assert len(serialize_policy(world, "native")) >= len(serialize_policy(world, "mirror")) - 3


def test_connectivity_aware_never_selects_disconnected_carrier():
    for seed in range(100):
        world = make_world(seed)
        for agent in policy_agents(world, "connectivity_aware"):
            assert agent is None or world.reachable[agent]


def test_central_posterior_not_worse_than_connectivity_code():
    row = evaluate(701, draws=512)
    assert row["central"]["nmse"] <= row["connectivity_aware"]["nmse"]
    assert row["connectivity_aware"]["disconnected_selection_violations"] == 0


def test_fixed_payload_lengths_charge_schema_and_view_addresses():
    world = make_world(733)
    schema_bytes = 14  # 2 byte header + 4 coordinate flags + 8 reliability classes
    assert len(serialize_policy(world, "native")) == schema_bytes + 3
    assert len(serialize_policy(world, "mirror")) == schema_bytes + 5
