# utils.py
import numpy as np
from signals import DiscreteSignal, ContinuousSignal

def parse_discrete_signal(values_str: str, start_str: str) -> DiscreteSignal:
    try:
        vals = np.array([float(v.strip()) for v in values_str.split(',')])
    except Exception:
        raise ValueError("Discrete values must be comma-separated numbers.")
    try:
        start = int(start_str)
    except Exception:
        raise ValueError("Start index must be an integer.")
    if vals.size == 0:
        raise ValueError("Signal vector cannot be empty.")
    return DiscreteSignal(vals, start)

def parse_continuous_signal(kind: str, t0_str: str, t1_str: str, A_str: str) -> ContinuousSignal:
    try:
        t0 = float(t0_str)
        t1 = float(t1_str)
        A = float(A_str)
    except Exception:
        raise ValueError("t0, t1, and amplitude must be numeric.")
    if t1 <= t0:
        raise ValueError("t1 must be greater than t0.")
    return ContinuousSignal(kind, t0, t1, A)
