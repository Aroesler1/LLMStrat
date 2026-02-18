from quantaalpha.factors.regulator.factor_regulator import FactorRegulator


def test_is_parsable_handles_missing_parenthesis_with_sanitize() -> None:
    reg = FactorRegulator()
    # one missing ')' at end
    expr = "LOG($close/(DELAY($close,1)+1e-8)"
    assert reg.is_parsable(expr) is True


def test_is_parsable_rejects_deep_nested_ternary_without_crashing() -> None:
    reg = FactorRegulator()
    expr = (
        "(TS_SUM($volume,1)>=0.05*DELAY(TS_MEAN($volume,20),1))?(1):"
        "((TS_SUM($volume,2)>=0.05*DELAY(TS_MEAN($volume,20),1))?(2):"
        "((TS_SUM($volume,3)>=0.05*DELAY(TS_MEAN($volume,20),1))?(3):"
        "((TS_SUM($volume,4)>=0.05*DELAY(TS_MEAN($volume,20),1))?(4):"
        "((TS_SUM($volume,5)>=0.05*DELAY(TS_MEAN($volume,20),1))?(5):(6)))))"
    )
    assert reg.is_parsable(expr) is False


def test_is_parsable_handles_complex_expression_without_exception() -> None:
    reg = FactorRegulator()
    expr = (
        "REGRESI(FILTER((SUMIF($volume,10,$close>$open)-SUMIF($volume,10,$close<$open))/"
        "(SUMIF($volume,10,$close>$open)+SUMIF($volume,10,$close<$open)+1e-8),$close==$close),"
        "FILTER(((($close/DELAY($close,1))-1),(($high-$low)/($close+1e-8))),$close==$close),60)"
    )
    # Current behavior may be True/False depending on parser internals; key is no crash.
    assert isinstance(reg.is_parsable(expr), bool)
