import mlflow
from mflowy.builtin_plugins.cross_validation.types import X_y
from mflowy.driver.context import Context
from mflowy.driver.handler import Handler


def log_X_y(ctx: Context, next: Handler):
    res: X_y = next(ctx)
    X, y = res
    mlflow.log_metrics(
        {
            "input_dim": X.shape[1],
            "output_dim": y.shape[1],
        }
    )
