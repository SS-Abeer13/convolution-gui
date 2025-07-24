# signals.py
import numpy as np

class DiscreteSignal:
    def __init__(self, values: np.ndarray, start: int):
        self.values = values
        self.start = start

    def time_indices(self):
        return np.arange(self.start, self.start + len(self.values))

class ContinuousSignal:
    def __init__(self, kind: str, t0: float, t1: float, A: float, num=1000):
        self.kind = kind
        self.t0 = t0
        self.t1 = t1
        self.A = A
        self.num = num

    def samples(self):
        # For signals with horizontal tails, extend domain
        needs_tails = self.kind in ("step", "rectangular", "ramp")
        if needs_tails:
            span = self.t1 - self.t0
            t = np.linspace(self.t0 - span, self.t1 + span, self.num)
        else:
            t = np.linspace(self.t0, self.t1, self.num)
        # Generate waveform
        if self.kind == "impulse":
            x = np.zeros_like(t)
            idx = np.argmin(np.abs(t - self.t0))
            x[idx] = self.A
        elif self.kind == "step":
            x = self.A * (t >= self.t0).astype(float)
        elif self.kind == "rectangular":
            x = np.zeros_like(t)
            mask = (t >= self.t0) & (t <= self.t1)
            x[mask] = self.A
        elif self.kind == "triangular":
            mid = (self.t0 + self.t1) / 2
            x = np.maximum(1 - np.abs(t - mid) / (mid - self.t0), 0) * self.A
        elif self.kind == "sawtooth":
            period = self.t1 - self.t0
            x = self.A * ((t - self.t0) % period) / period
        elif self.kind == "ramp":
            # ramp up from 0 at t0 to A at t1, flat afterward
            x = np.zeros_like(t)
            ramp_mask = (t >= self.t0) & (t <= self.t1)
            x[ramp_mask] = self.A * (t[ramp_mask] - self.t0) / (self.t1 - self.t0)
            x[t > self.t1] = self.A
        else:
            raise ValueError(f"Unknown continuous kind: {self.kind}")
        return t, x
