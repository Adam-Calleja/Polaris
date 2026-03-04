from polaris_rag.evaluation.evaluator import ConcurrencyTrial, _select_best_trial


def test_select_best_trial_prefers_throughput_within_threshold() -> None:
    trials = [
        ConcurrencyTrial(workers=2, duration_seconds=5.0, failure_rate=0.0, throughput=20.0, failures=0, total_scores=100),
        ConcurrencyTrial(workers=4, duration_seconds=3.0, failure_rate=0.01, throughput=33.0, failures=1, total_scores=100),
        ConcurrencyTrial(workers=8, duration_seconds=2.0, failure_rate=0.05, throughput=50.0, failures=5, total_scores=100),
    ]

    selected = _select_best_trial(trials, failure_threshold=0.02)
    assert selected == 4


def test_select_best_trial_falls_back_to_lowest_failure_rate() -> None:
    trials = [
        ConcurrencyTrial(workers=2, duration_seconds=5.0, failure_rate=0.10, throughput=20.0, failures=10, total_scores=100),
        ConcurrencyTrial(workers=4, duration_seconds=2.0, failure_rate=0.10, throughput=30.0, failures=10, total_scores=100),
        ConcurrencyTrial(workers=8, duration_seconds=1.0, failure_rate=0.20, throughput=40.0, failures=20, total_scores=100),
    ]

    selected = _select_best_trial(trials, failure_threshold=0.02)
    assert selected == 4
