from llm.prompts.hashtags_prompt import parse_hashtags
from llm.prompts.titles_prompt import parse_titles


def test_parse_titles_numbered_list():
    raw = (
        "1. Кот, который не боится ничего \n"
        "2) ОН ПРЫГНУЛ ТУДА, КУДА НЕ ДОЛЖЕН\n"
        "3 - Это было неожиданно...\n"
        "4. Просто обычный день кота\n"
    )
    titles = parse_titles(raw, expected_count=10)
    assert len(titles) == 4
    assert titles[2] == "Это было неожиданно..."


def test_parse_titles_fallback_without_numbering():
    raw = "Кот прыгнул\nОн упал\nВсем смешно"
    assert parse_titles(raw) == ["Кот прыгнул", "Он упал", "Всем смешно"]


def test_parse_hashtags_deduplicates_case_insensitive():
    raw = "Вот теги: #cats #CatsOfInstagram #cats #funny #catfail #Funny"
    tags = parse_hashtags(raw, max_count=30)
    assert tags == ["#cats", "#CatsOfInstagram", "#funny", "#catfail"]


def test_parse_hashtags_respects_max_count():
    raw = " ".join(f"#tag{i}" for i in range(50))
    assert len(parse_hashtags(raw, max_count=30)) == 30
