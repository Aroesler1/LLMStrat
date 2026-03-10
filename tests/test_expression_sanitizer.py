from quantaalpha_us.factors.expression_sanitizer import ExpressionSanitizer


def test_valid_expression_passes() -> None:
    sanitizer = ExpressionSanitizer()
    result = sanitizer.sanitize("TS_MEAN($close, 20) / (TS_STD($close, 20) + 1e-8)")
    assert result.valid
    assert result.errors == []


def test_blocked_token_rejected() -> None:
    sanitizer = ExpressionSanitizer()
    result = sanitizer.sanitize("import os; TS_MEAN($close, 20)")
    assert not result.valid
    assert any("Blocked token" in e for e in result.errors)


def test_unknown_function_rejected() -> None:
    sanitizer = ExpressionSanitizer()
    result = sanitizer.sanitize("HACK_FUNC($close, 10)")
    assert not result.valid
    assert any("Unknown function" in e for e in result.errors)


def test_length_rejected() -> None:
    sanitizer = ExpressionSanitizer(max_expression_length=10)
    result = sanitizer.sanitize("TS_MEAN($close, 20)")
    assert not result.valid
    assert any("Expression too long" in e for e in result.errors)
