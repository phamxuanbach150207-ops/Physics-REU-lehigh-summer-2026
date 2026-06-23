import numpy as np
import matplotlib.pyplot as plt
import math
from Tool_box import *

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

data = get_data("F:\Physics REU Lehigh 2026\Data\june 8\june8_orient_1.xlsx")
y = extract_column(data,1)
y = normalize(truncate_to_peak(y))
t = time_converter(y, 0.016)

y_field = extract_column(data,2)
y_field = normalize(truncate_to_peak(y_field))
t_field = time_converter(y_field, 0.016)

zf_integral = integrated_yield_in_window(t,y,0,10)
field_integral = integrated_yield_in_window(t_field,y_field,0,10)

print(zf_integral)
print(field_integral)

print(field_integral/zf_integral)

plt.plot(t,y)
plt.plot(t_field,y_field)
plt.xlim(0,15)
plt.yscale('log')
plt.grid(alpha=0.3)
plt.xlabel('t [ns]')
plt.ylabel('PL(t) (normalized)')
plt.show()
print(data)
