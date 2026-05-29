from solution import dedupe_keep_order


def test_dedupe_keep_order():
    assert dedupe_keep_order(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_empty():
    assert dedupe_keep_order([]) == []
