from collections.abc import Iterator

import numpy as np
import pandas as pd

X_idx = 0
y_idx = 1
type X_y = tuple[pd.DataFrame, pd.DataFrame]
type Indices = tuple[np.ndarray, np.ndarray | None, np.ndarray]
type Dataset = tuple[X_y, X_y | None, X_y]
type DatasetLoader = Iterator[Dataset]
