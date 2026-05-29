from solution import moving_average


def test_window_three():
    assert moving_average([1, 2, 3, 4, 5], 3) == [2.0, 3.0, 4.0]


def test_invalid_window():
    assert moving_average([1, 2], 3) == []
