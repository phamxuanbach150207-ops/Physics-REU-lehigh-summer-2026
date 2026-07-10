import numpy as np
import matplotlib.pyplot as plt
import math
from DataExtraction_Tools import *

DATA_PATH = r"C:\Users\bap229\Desktop\Data\june8_orient_1.xlsx"

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


def subtract_offset(y, offset):
    yy = []
    for i in y:
        yy.append(i - offset)
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


def truncate_time_window(t, y, t_min, t_max):
    """
    Return truncated t and y arrays with t in [t_min, t_max].
    """
    t = np.asarray(t)
    y = np.asarray(y)
    mask = (t >= t_min) & (t <= t_max)
    return t[mask], y[mask]


def divide_traces(t1, y1, t2, y2):
    """
    Divide y1/y2 assuming identical time axes.
    Returns (t, ratio).
    """
    t1 = np.asarray(t1)
    t2 = np.asarray(t2)
    if len(t1) != len(t2):
        raise ValueError("Time arrays have different lengths")
    if not np.allclose(t1, t2):
        raise ValueError("Time axes do not match")
    return t1, np.asarray(y1) / np.asarray(y2)


data = get_data(DATA_PATH)

y_zero = extract_column(data, 1)
y_zero = normalize(truncate_to_peak(y_zero))
t_zero = time_converter(y_zero, 0.016)
int_zero = integrated_yield_in_window(t_zero, y_zero, 0, 10)

ratios = []
for i in range(1, 19):
    y = extract_column(data, i)
    y = normalize(truncate_to_peak(y))
    t = time_converter(y, 0.016)
    integral = integrated_yield_in_window(t, y, 0, 10)
    rat = integral / int_zero
    ratios.append(rat)

field_strengths = [0, 40, 100, 200, 400, 600, 800, 1000, 1500, 2000, 3000, 4000,
                    5000, 6000, 7000, 8000, 9000, 10000]

plt.scatter(field_strengths, ratios)
plt.plot(field_strengths, ratios, linewidth=0.75)
plt.axhline(y=1, color='black', linewidth=1)
plt.xlabel('Field Strength [Gauss]')
plt.ylabel('PL(t) Integration Ratio [0-10 ns]')
plt.ylim(0.9, 1.2)
plt.grid(alpha=0.3)
plt.show()
