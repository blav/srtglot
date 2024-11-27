from srtglot.fallback import fit_fragments_count
from srtglot.languages import LanguageTokenization


def test_fit_fragments_count_space():
    assert fit_fragments_count(
        LanguageTokenization.SPACE, 2, ["Bonjour", "le", "monde"]
    ) == ["Bonjour le", "monde"]

    assert fit_fragments_count(
        LanguageTokenization.SPACE, 1, ["Bonjour", "le", "monde"]
    ) == ["Bonjour le monde"]

    assert fit_fragments_count(
        LanguageTokenization.SPACE, 4, ["Bonjour", "le pot", "de demain"]
    ) == ["Bonjour", "le pot", "de", "demain"]

    assert fit_fragments_count(
        LanguageTokenization.SPACE, 5, ["Bonjour", "le pot", "de demain"]
    ) == ["Bonjour", "le", "pot", "de", "demain"]

    assert fit_fragments_count(
        LanguageTokenization.SPACE, 6, ["Bonjour", "le pot", "de demain"]
    ) == ["Bonjour", "le", "pot", "de", "demain", ""]


def test_fit_fragments_count_cjk():
    assert fit_fragments_count(
        LanguageTokenization.CJK, 2, ["A", "B", "C"]
    ) == ["AB", "C"]

    assert fit_fragments_count(
        LanguageTokenization.CJK, 1, ["A", "B", "C"]
    ) == ["ABC"]

    assert fit_fragments_count(
        LanguageTokenization.CJK, 4, ["A", "BC", "DE"]
    ) == ["A", "B", "C", "DE"]

    assert fit_fragments_count(
        LanguageTokenization.CJK, 5, ["A", "BC", "DE"]
    ) == ["A", "B", "C", "D", "E"]

    assert fit_fragments_count(
        LanguageTokenization.CJK, 6, ["A", "BC", "DE"]

    ) == ["A", "B", "C", "D", "E", ""]

