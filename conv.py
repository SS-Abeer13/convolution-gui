import numpy as np
from signals import DiscreteSignal, ContinuousSignal


def discrete_convolve(x_sig: DiscreteSignal, h_sig: DiscreteSignal) -> DiscreteSignal:
    """
    Perform discrete convolution and return a new DiscreteSignal.
    """
    y_vals = np.convolve(x_sig.values, h_sig.values)
    y_start = x_sig.start + h_sig.start
    return DiscreteSignal(y_vals, y_start)


def continuous_convolve(x_sig: ContinuousSignal, h_sig: ContinuousSignal):
    """
    Perform continuous convolution via direct sum approximation and return (t, y).
    Time vector t starts at t0_x + t0_h and increments by dt.
    """
    tx, x = x_sig.samples()
    th, h = h_sig.samples()
    dt = tx[1] - tx[0]
    # direct convolution
    y = np.convolve(x, h) * dt
    # build time axis
    t_start = tx[0] + th[0]
    t = t_start + np.arange(len(y)) * dt
    return t, y
