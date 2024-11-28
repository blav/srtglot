from srtglot.completions import parse_completions


def test_should_parse_completions():
    content = """[sentence 1]
    Hello, world!
    Bonjour, tout le monde!

    [sentence 2]
    Goodbye, world!
    Au revoir, tout le monde!
    """
    completions = parse_completions([], content)
    assert completions == [
        ["Hello, world!", "Bonjour, tout le monde!"],
        ["Goodbye, world!", "Au revoir, tout le monde!"],
    ]
