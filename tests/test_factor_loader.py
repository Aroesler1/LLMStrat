import json

import pytest

from quantaalpha.backtest.factor_loader import FactorLoader


def test_factor_loader_accepts_schema_valid_factor_json(tmp_path):
    factor_file = tmp_path / "factors_valid.json"
    factor_file.write_text(
        json.dumps(
            {
                "factors": {
                    "f1": {
                        "factor_name": "F1",
                        "factor_expression": "$close/Ref($close,1)-1",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    loader = FactorLoader(
        {
            "factor_source": {
                "type": "custom",
                "custom": {"json_files": [str(factor_file)]},
            }
        }
    )
    qlib_factors, custom_factors = loader.load_factors()

    assert qlib_factors == {}
    assert len(custom_factors) == 1
    assert custom_factors[0]["factor_name"] == "F1"


def test_factor_loader_accepts_object_cache_location(tmp_path):
    factor_file = tmp_path / "factors_obj_cache.json"
    factor_file.write_text(
        json.dumps(
            {
                "factors": {
                    "f1": {
                        "factor_name": "F1",
                        "factor_expression": "$close/Ref($close,1)-1",
                        "cache_location": {
                            "workspace_suffix": "exp_20260218_015818",
                            "workspace_path": "/tmp/workspace",
                            "factor_dir": "abc123",
                            "result_h5_path": "/tmp/workspace/abc123/result.h5",
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    loader = FactorLoader(
        {
            "factor_source": {
                "type": "custom",
                "custom": {"json_files": [str(factor_file)]},
            }
        }
    )
    _, custom_factors = loader.load_factors()

    assert custom_factors[0]["cache_location"]["result_h5_path"].endswith("result.h5")


def test_factor_loader_normalizes_legacy_string_cache_location(tmp_path):
    factor_file = tmp_path / "factors_str_cache.json"
    factor_file.write_text(
        json.dumps(
            {
                "factors": {
                    "f1": {
                        "factor_name": "F1",
                        "factor_expression": "$close/Ref($close,1)-1",
                        "cache_location": "/tmp/legacy/result.h5",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    loader = FactorLoader(
        {
            "factor_source": {
                "type": "custom",
                "custom": {"json_files": [str(factor_file)]},
            }
        }
    )
    _, custom_factors = loader.load_factors()

    assert custom_factors[0]["cache_location"] == {"result_h5_path": "/tmp/legacy/result.h5"}


def test_factor_loader_rejects_schema_invalid_factor_json(tmp_path):
    factor_file = tmp_path / "factors_invalid.json"
    factor_file.write_text(json.dumps({"not_factors": []}), encoding="utf-8")
    loader = FactorLoader(
        {
            "factor_source": {
                "type": "custom",
                "custom": {"json_files": [str(factor_file)]},
            }
        }
    )

    with pytest.raises(ValueError, match="Invalid factor JSON schema"):
        loader.load_factors()
