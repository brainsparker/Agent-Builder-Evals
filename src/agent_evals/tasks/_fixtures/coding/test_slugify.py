from solution import slugify


def test_slugify_basic():
    assert slugify("Hello, Agent Builder!") == "hello-agent-builder"


def test_slugify_collapse():
    assert slugify("  A---B__C  ") == "a-b-c"
