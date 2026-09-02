from fqc.local_bundle import best_bundle_under_budget, enumerate_bundle


def test_joint_layout_precision_move_can_beat_each_coordinate_alone() -> None:
    # Mechanism-preserving reconstruction of E7, not the original hidden setup.
    per_leaf = [range(5) for _ in range(8)]

    base_dist = [4.0, 3.7, 3.3, 2.9, 2.2, 1.9, 1.5, 1.2]
    gain_23 = [0.30, 0.26, 0.22, 0.19464, 0.0, 0.0, 0.0, 0.0]

    def distortion(states: tuple[int, ...]) -> float:
        total = 0.0
        for i, s in enumerate(states):
            total += base_dist[i] + max(0, 2 - s) * 2.0
            if s >= 3:
                total -= gain_23[i]
        return total

    def bit_cost(layout: str, states: tuple[int, ...]) -> int:
        bits = 24 + sum(max(0, s - 2) * 2 for s in states)
        if layout == "aligned":
            if states[:4] == (3, 3, 3, 3) and states[4:] == (2, 2, 2, 2):
                bits -= 8
        return bits

    assignments = enumerate_bundle(
        per_leaf,
        layouts=["crossed", "aligned"],
        bit_cost=bit_cost,
        distortion=distortion,
    )
    budget = 24
    incumbent = next(
        a for a in assignments if a.layout == "crossed" and a.states == (2,) * 8
    )

    crossed_best = best_bundle_under_budget(
        (a for a in assignments if a.layout == "crossed"), budget
    )
    assert crossed_best is not None
    assert crossed_best.distortion == incumbent.distortion

    aligned_same_repr = next(
        a for a in assignments if a.layout == "aligned" and a.states == (2,) * 8
    )
    assert aligned_same_repr.distortion == incumbent.distortion

    joint_best = best_bundle_under_budget(assignments, budget)
    assert joint_best is not None
    assert joint_best.layout == "aligned"
    assert joint_best.states == (3, 3, 3, 3, 2, 2, 2, 2)
    assert joint_best.bits == 24
    assert joint_best.distortion < incumbent.distortion
