import os
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import numpy as np
import scipy
import math
from scipy.optimize import curve_fit
import csaps
from scipy.signal import hilbert, lombscargle, savgol_filter


def get_data(path_name):
    mydata = os.path.abspath(path_name)
    datafile = pd.read_excel(mydata, header=None)
    return datafile


def get_last_elements(lst, a):
    if a >= len(lst):
        return lst
    return lst[-a:]


def gaussian_filter(x, y, sigma):
    return gaussian_filter1d(y, sigma)


def truncate_time_series(x, y, start, end):
    truncated_x, truncated_y = [], []
    for time, value in zip(x, y):
        if start <= time <= end:
            truncated_x.append(time)
            truncated_y.append(value)
    return truncated_x, truncated_y


def spliner(x, y, smooth):
    sp = csaps.CubicSmoothingSpline(x, y, smooth=smooth)
    return sp(x)


def ratio(y1, y2):
    return [(i / j) - 1 for i, j in zip(y1, y2)]


def hilbert_envelope(y):
    return np.abs(hilbert(y))


def fft(x, y, f_min=0.01, f_max=10.0, num_points=2000):
    x = np.array(x)
    y = np.array(y)
    f = np.linspace(f_min, f_max, num_points)
    angular_freqs = 2 * np.pi * f
    power = lombscargle(x, y, angular_freqs, normalize=True)
    peak_freq = f[np.argmax(power)]
    print('FFT Frequency: ' + str(peak_freq))
    return f.tolist(), power.tolist()


def trim_and_sum(x, *y_lists):
    min_len = min(len(x), *(len(y) for y in y_lists))
    x_trimmed = x[:min_len]
    y_trimmed = [y[:min_len] for y in y_lists]
    y_sum = [sum(values) for values in zip(*y_trimmed)]
    return x_trimmed, y_sum


def smooth_bump_region(x, y, x_min, x_max, window_length=11, polyorder=3):
    x, y = np.array(x), np.array(y)
    y_smoothed = y.copy()
    bump_mask = (x >= x_min) & (x <= x_max)
    y_smoothed[bump_mask] = savgol_filter(y[bump_mask], window_length, polyorder)
    return y_smoothed


def flatten_bump(x, y, x_min, x_max):
    x, y = np.array(x), np.array(y)
    y_fixed = y.copy()
    bump_mask = (x >= x_min) & (x <= x_max)
    indices = np.where(bump_mask)[0]
    if len(indices) == 0:
        raise ValueError("No data points in the bump region.")
    left_idx = indices[0] - 1
    right_idx = indices[-1] + 1
    if left_idx < 0 or right_idx >= len(x):
        raise ValueError("Bump too close to edge for interpolation.")
    y_fixed[indices] = np.interp(x[indices], [x[left_idx], x[right_idx]], [y[left_idx], y[right_idx]])
    return y_fixed


def suppress_bump_gently(x, y, x_min, x_max, window_length=7, polyorder=2, blend=0.8):
    x, y = np.array(x), np.array(y)
    y_fixed = y.copy()
    bump_mask = (x >= x_min) & (x <= x_max)
    indices = np.where(bump_mask)[0]
    if len(indices) < window_length:
        window_length = len(indices) if len(indices) % 2 == 1 else len(indices) - 1
    if window_length < 3:
        return y_fixed
    y_smooth = savgol_filter(y[indices], window_length, polyorder)
    y_fixed[indices] = (1 - blend) * y[indices] + blend * y_smooth
    return y_fixed


def scale_bump_region(x, y, x_min, x_max, factor):
    x, y = np.array(x), np.array(y)
    y_scaled = y.copy()
    y_scaled[(x >= x_min) & (x <= x_max)] *= factor
    return y_scaled


def exponential_fit(x, y):
    def exponential(x, a, k):
        return a * np.exp(-x / k)
    x, y = np.array(x), np.array(y)
    popt, pcov = curve_fit(exponential, x, y, p0=[0.05, 10], maxfev=50000)
    y_fit = exponential(x, *popt)
    perr = np.sqrt(np.diag(pcov))
    print(f"Time Constant (k): {popt[1]:.5f} ± {perr[1]:.5f}")
    print(f"Pre-Factor (a): {popt[0]:.5f} ± {perr[0]:.5f}")
    return y_fit


def custom_fit(x, y):
    def custom(x, a, b, c, d, f, g):
        return a * np.exp(-x / b) + c * np.exp(-x / d) + f * g * x
    popt, _ = scipy.optimize.curve_fit(custom, x, y, p0=[1, 1, 1, 1, 1, 1], maxfev=100000)
    return [custom(i, *popt) for i in x]


def extract_column(dataframe, n):
    """Extract column n (1-indexed) from dataframe as a list."""
    return list(dataframe.iloc[:, n - 1])


def truncate_to_peak(y):
    """Return the decay portion starting just after the peak."""
    peak_idx = y.index(max(y))
    return y[peak_idx + 1:]


def time_converter(y, dt):
    """Generate time axis with bin width dt (ns)."""
    return [i * dt for i in range(len(y))]


def normalize(y):
    if not y:
        return []
    peak = max(y)
    if peak == 0:
        return [0] * len(y)
    return [v / peak for v in y]


def subtract_offset(y, offset):
    """Subtract offset and clip negatives to zero."""  # BUG FIX: was returning yy (unclipped)
    return [max(v - offset, 0) for v in y]


def integrated_yield_in_window(t, y, t1, t2):
    """Sum counts in time window [t1, t2] ns."""
    t, y = np.asarray(t), np.asarray(y)
    return float(np.sum(y[(t >= t1) & (t <= t2)]))


def log_bin(x, y, num_bins):
    del x[0]
    del y[0]
    x, y = np.array(x), np.array(y)
    if np.any(x <= 0):
        raise ValueError("All x values must be positive for logarithmic binning.")
    bins = np.logspace(np.log10(min(x)), np.log10(max(x)), num_bins + 1)
    binned_x, binned_y = [], []
    for i in range(num_bins):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        if np.any(mask):
            binned_x.append(np.mean(x[mask]))
            binned_y.append(np.mean(y[mask]))
    return np.array(binned_x), np.array(binned_y)


def accumulated_count(x, y):
    x, y = np.array(x), np.array(y)
    return x, np.cumsum(y)
