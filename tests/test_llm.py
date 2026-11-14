from atlas.llm import normalise_stream
from atlas.pipeline import extract_citations


def run(*deltas: str) -> str:
    return "".join(normalise_stream(deltas))


def test_fullwidth_citations_become_brackets_even_when_split():
    text = run("Employees get 24 days【", "1", "】 and carry over 5【2†L3", "-L5】.")
    assert text == "Employees get 24 days[1] and carry over 5[2]."
    assert extract_citations(text, 5) == [1, 2]


def test_narrow_spaces_hyphens_and_bold_are_made_plain():
    assert run("Full‑time staff spray **22 hectares", "*", "* per hour [1].") == "Full-time staff spray 22 hectares per hour [1]."


def test_text_without_special_characters_passes_through_unchanged():
    deltas = ["The limit ", "is $300 ", "per night [1]."]
    assert list(normalise_stream(deltas)) == deltas


def test_an_unclosed_bracket_is_not_held_forever():
    assert run("A stray 【 in a long sentence that never closes it.") == "A stray 【 in a long sentence that never closes it."


def test_no_space_is_left_before_a_percent_sign():
    assert run("matches 100\u202f", "% up to 4\u00a0%", " of salary [1].") == "matches 100% up to 4% of salary [1]."
