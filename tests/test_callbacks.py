import pytest
import pandas as pd
from src.callbacks import CSVTrackingCallback


class MockTrainerState:
    global_step = 10
    epoch = 0.5


def test_csv_tracking_callback(tmp_path):
    train_csv = tmp_path / "train_log.csv"
    eval_csv = tmp_path / "eval_log.csv"

    cb = CSVTrackingCallback(train_csv=str(train_csv), eval_csv=str(eval_csv))

    assert train_csv.exists()
    assert eval_csv.exists()

    state = MockTrainerState()
    # Test on_log
    cb.on_log(args=None, state=state, control=None, logs={"loss": 1.234, "learning_rate": 0.0001, "grad_norm": 0.5})
    df_train = pd.read_csv(train_csv)
    assert len(df_train) == 1
    assert df_train.iloc[0]["step"] == 10
    assert abs(df_train.iloc[0]["loss"] - 1.234) < 1e-4

    # Test on_evaluate
    cb.on_evaluate(args=None, state=state, control=None, metrics={"eval_loss": 0.987, "eval_runtime": 12.3})
    df_eval = pd.read_csv(eval_csv)
    assert len(df_eval) == 1
    assert df_eval.iloc[0]["step"] == 10
    assert abs(df_eval.iloc[0]["eval_loss"] - 0.987) < 1e-4
