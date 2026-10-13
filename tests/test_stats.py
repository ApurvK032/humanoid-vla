import pytest

from humanoid_vla.utils.stats import wilson_interval


def test_wilson_interval_known_value() -> None:
    low, high = wilson_interval(81, 100)
    assert low == pytest.approx(0.7222, abs=1e-4)
    assert high == pytest.approx(0.8749, abs=1e-4)


def test_wilson_interval_at_the_extremes_stays_in_range() -> None:
    low, high = wilson_interval(100, 100)
    assert 0.96 < low < 1.0
    assert high == 1.0
    low, high = wilson_interval(0, 50)
    assert low == 0.0
    assert 0.0 < high < 0.08


@pytest.mark.parametrize(("successes", "trials"), [(1, 0), (-1, 10), (11, 10)])
def test_wilson_interval_rejects_invalid_counts(successes: int, trials: int) -> None:
    with pytest.raises(ValueError, match="must be"):
        wilson_interval(successes, trials)
