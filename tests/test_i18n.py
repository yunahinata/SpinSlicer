from i18n import language, set_language, tr


def test_english_translation_is_available_and_russian_can_be_restored() -> None:
    original = language()
    try:
        set_language("en")
        assert tr("Настройки процесса") == "Process settings"
        set_language("ru")
        assert tr("Настройки процесса") == "Настройки процесса"
    finally:
        set_language(original)
