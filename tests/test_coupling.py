from fqc.coupling import ScalarBlockMetric, is_psd_2x2


def test_diagonal_model_can_underestimate_true_cross_block_distortion() -> None:
    metric = ScalarBlockMetric(((2.0, 1.5), (1.5, 2.0)))
    assert is_psd_2x2(metric)
    error = (1.0, 1.0)
    assert metric.diagonal_approx(error) == 4.0
    assert metric.quadratic(error) == 7.0


def test_row_sum_majorizer_is_safe_for_positive_and_negative_cross_terms() -> None:
    for cross in (1.5, -1.5):
        metric = ScalarBlockMetric(((2.0, cross), (cross, 2.0)))
        assert is_psd_2x2(metric)
        for error in ((1.0, 1.0), (1.0, -1.0), (0.3, 2.1), (-2.0, 0.7)):
            assert metric.safe_row_sum_majorizer(error) + 1e-12 >= metric.quadratic(error)
