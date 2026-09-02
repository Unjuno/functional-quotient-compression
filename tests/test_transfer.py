from fqc.transfer import MaskState, conditional_frontier_transfer, direct_rotated_optimum


def test_conditional_transfer_matches_direct_description_only_rotation() -> None:
    states = []
    for mask in range(1 << 8):
        pop = mask.bit_count()
        bits = 12 + pop + (2 if mask & 0b11110000 else 0)
        distortion = float(20 - 0.6 * pop + 0.01 * (mask % 7))
        states.append(MaskState(mask, bits, distortion))

    local = lambda mask: mask & 0b111
    delta_map = {s: ((s.bit_count() % 3) - 1) for s in range(8)}
    delta_sig = lambda s: delta_map[s]
    delta_mask = lambda mask: delta_sig(local(mask))

    for budget in range(12, 24):
        direct = direct_rotated_optimum(states, budget_bits=budget, delta_bits=delta_mask)
        transferred = conditional_frontier_transfer(
            states,
            budget_bits=budget,
            local_signature=local,
            delta_for_signature=delta_sig,
        )
        assert (direct is None) == (transferred is None)
        if direct is not None and transferred is not None:
            assert direct.distortion == transferred.distortion
            assert direct.bits + delta_mask(direct.mask) <= budget
            assert transferred.bits + delta_mask(transferred.mask) <= budget
