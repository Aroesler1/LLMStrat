import pytest

from quantaalpha.factors.coder.expr_parser import check_for_invalid_operators, parse_expression


def test_operator_check_allows_comma_followed_by_unary_minus() -> None:
    expr = "DELTA($close,5)*MAX(MIN(TS_ZSCORE($volume,20),2),-2)"
    check_for_invalid_operators(expr)


def test_parse_expression_allows_negative_constant_argument() -> None:
    expr = "DELTA($close,5)*MAX(MIN(TS_ZSCORE($volume,20),2),-2)"
    parsed = parse_expression(expr)
    assert isinstance(parsed, str)
    assert "MAX(" in parsed


def test_operator_check_rejects_invalid_operator_cluster() -> None:
    with pytest.raises(Exception):
        check_for_invalid_operators("A ~~ B")
