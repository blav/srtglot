from srtglot.context import TranslatorError


def test_should_have_batch_and_completions_fields():
    sentences = []
    translations = []
    error = TranslatorError(sentences, translations, "message")
    assert error.batch is sentences
    assert error.completions is translations
    assert str(error) == "message"