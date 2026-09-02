from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ScalarBlockMetric:
    """Small scalar-block quadratic metric used for exact coupling tests."""

    matrix: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        n = len(self.matrix)
        if n == 0 or any(len(row) != n for row in self.matrix):
            raise ValueError("matrix must be non-empty and square")
        for i in range(n):
            for j in range(n):
                if abs(self.matrix[i][j] - self.matrix[j][i]) > 1e-12:
                    raise ValueError("matrix must be symmetric")

    def quadratic(self, error: Sequence[float]) -> float:
        if len(error) != len(self.matrix):
            raise ValueError("dimension mismatch")
        return sum(
            error[i] * self.matrix[i][j] * error[j]
            for i in range(len(error))
            for j in range(len(error))
        )

    def diagonal_approx(self, error: Sequence[float]) -> float:
        return sum(self.matrix[i][i] * error[i] * error[i] for i in range(len(error)))

    def safe_row_sum_majorizer(self, error: Sequence[float]) -> float:
        """Return the scalar-block version of G_aa + r_a I majorization.

        For scalar blocks, c_ab = |G_ab| and r_a = sum_{b!=a}|G_ab|.
        The inequality follows from 2|x_i x_j| <= x_i^2 + x_j^2.
        """
        total = 0.0
        for i, value in enumerate(error):
            radius = sum(abs(self.matrix[i][j]) for j in range(len(error)) if j != i)
            total += (self.matrix[i][i] + radius) * value * value
        return total


def is_psd_2x2(metric: ScalarBlockMetric, *, tol: float = 1e-12) -> bool:
    """Exact PSD check for the two-block toy case."""
    if len(metric.matrix) != 2:
        raise ValueError("is_psd_2x2 only supports 2x2 matrices")
    a, b = metric.matrix[0]
    _, d = metric.matrix[1]
    return a >= -tol and d >= -tol and a * d - b * b >= -tol
