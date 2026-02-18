import logging
import contextlib
import io
import warnings
from pathlib import Path

import pandas as pd
with contextlib.redirect_stderr(io.StringIO()):
    import qlib

warnings.filterwarnings(
    "ignore",
    message="Mean of empty slice",
    category=RuntimeWarning,
)
warnings.filterwarnings(
    "ignore",
    message="A value is trying to be set on a copy of a slice from a DataFrame",
    category=Warning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*Gym has been unmaintained since 2022.*",
    category=UserWarning,
)

qlib.init(logging_level=40)
logging.getLogger("qlib.BaseExecutor").setLevel(logging.ERROR)

from qlib.workflow import R


def _latest_recorder():
    latest = None
    latest_exp = None
    for exp_name in R.list_experiments():
        for recorder_id in R.list_recorders(experiment_name=exp_name):
            if recorder_id is None:
                continue
            recorder = R.get_recorder(recorder_id=recorder_id, experiment_name=exp_name)
            end_time = recorder.info.get("end_time")
            if end_time is None:
                continue
            if latest is None or end_time > latest.info.get("end_time"):
                latest = recorder
                latest_exp = exp_name
    return latest_exp, latest


def main() -> None:
    experiment_name, recorder = _latest_recorder()
    if recorder is None:
        print("No recorders found")
        return

    print(f"Latest recorder in experiment '{experiment_name}': {recorder}")
    metrics = pd.Series(recorder.list_metrics())
    output_path = Path(__file__).resolve().parent / "qlib_res.csv"
    metrics.to_csv(output_path)
    print(f"Output has been saved to {output_path}")

    ret_data_frame = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    ret_data_frame.to_pickle("ret.pkl")


if __name__ == "__main__":
    main()
