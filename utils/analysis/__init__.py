"""Dataset and time-series analyses for the accompanying notebooks."""
from .result_io import load_result, lr_matrices
from .time_series import TimeSeriesAnalysis

__all__ = ["load_result", "lr_matrices", "TimeSeriesAnalysis"]
