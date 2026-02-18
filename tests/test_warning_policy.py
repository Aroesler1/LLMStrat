from quantaalpha.utils.warning_policy import (
    EXPECTED_PYTHONWARNING_RULES,
    build_pythonwarnings_filter,
)


def test_build_pythonwarnings_filter_appends_expected_rules() -> None:
    merged = build_pythonwarnings_filter("default::DeprecationWarning")
    for rule in EXPECTED_PYTHONWARNING_RULES:
        assert rule in merged
    assert "default::DeprecationWarning" in merged


def test_build_pythonwarnings_filter_deduplicates_rules() -> None:
    existing = ",".join(EXPECTED_PYTHONWARNING_RULES)
    merged = build_pythonwarnings_filter(existing)
    parts = [p.strip() for p in merged.split(",") if p.strip()]
    for rule in EXPECTED_PYTHONWARNING_RULES:
        assert parts.count(rule) == 1
