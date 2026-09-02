from fqc.validation import Proposal, validate_top_k


def test_surrogate_never_commits_without_exact_improvement() -> None:
    proposals = [
        Proposal("looks-best-but-bad", 0.0),
        Proposal("actually-good", 0.1),
        Proposal("other", 0.2),
    ]
    exact = {
        "looks-best-but-bad": 10.5,
        "actually-good": 8.0,
        "other": 9.5,
    }
    result = validate_top_k(
        proposals,
        k=2,
        exact_objective=lambda p: exact[p.name],
        incumbent_objective=10.0,
    )
    assert result is not None
    assert result.proposal.name == "actually-good"
    assert result.exact_objective == 8.0


def test_no_exact_improvement_means_no_commit() -> None:
    proposals = [Proposal("a", 0.0), Proposal("b", 0.1)]
    result = validate_top_k(
        proposals,
        k=2,
        exact_objective=lambda p: 10.0,
        incumbent_objective=10.0,
    )
    assert result is None
