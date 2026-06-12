import numpy as np
import matplotlib.pyplot as plt
import math
from DataExtraction_Tools import *

def baseline_before_peak_median(counts, skip_last=0):
    """
    Median of all points BEFORE the first max (peak).
    Optionally skip the last `skip_last` bins right before the peak
    to avoid including the rising edge.

    Returns: (baseline, peak_idx, peak_val)
    """
    if not counts:
        raise ValueError("counts list is empty")

    peak_val = max(counts)
    peak_idx = counts.index(peak_val)  # first occurrence of max

    # end index for baseline region (exclude last `skip_last` bins before peak)
    end_idx = max(0, peak_idx - max(0, skip_last))

    if end_idx == 0:
        # nothing before the peak after skipping → baseline 0.0
        return 0.0, peak_idx, peak_val

    base = np.average(counts[:end_idx])
    return float(base), peak_idx, peak_val
def subtract_offset(y,offset):
    yy = []
    for i in y:
        yy.append(i-offset)
    yyy = []
    for i in yy:
        if i < 0:
            yyy.append(0)
        else:
            yyy.append(i)
    return yy
def integrated_yield_in_window(t, y, t1, t2):
    """
    Sum y values for bins with t in [t1, t2] ns.
    (dt cancels in ratios if constant, so sum is fine.)
    """
    t = np.asarray(t)
    y = np.asarray(y)

    mask = (t >= t1) & (t <= t2)
    return float(np.sum(y[mask]))
def scale_after_time_np(t, y, t_cut, a):
    """
    NumPy version (faster for large arrays).
    """
    t = np.array(t)
    y = np.array(y)

    y_scaled = y.copy()
    y_scaled[t >= t_cut] *= a

    return y_scaled

data = get_data