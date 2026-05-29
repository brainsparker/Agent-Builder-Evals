from solution import reverse_words


def test_reverse_words_strips_and_reverses_each_word():
    assert reverse_words(" hello   world ") == "olleh dlrow"


def test_reverse_words_empty():
    assert reverse_words("") == ""
