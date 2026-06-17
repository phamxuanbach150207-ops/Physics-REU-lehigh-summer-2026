import os
from os import lstat
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import numpy as np
import scipy
import math
from scipy.optimize import curve_fit
import csaps
from scipy.signal import hilbert
from scipy.signal import lombscargle
from scipy.signal import savgol_filter

def get_data(path_name):
    mydata = os.path.abspath(path_name)
    datafile = pd.read_excel(mydata)
    return datafile

def get_last_elements(lst,a):
    if a >= len(lst):
        return lst
    else:
        return lst[-a:]

def gaussian_filter(x,y,sigma):
    filtered_y = gaussian_filter1d(y,sigma)
    return filtered_y

def truncate_time_series(x,y,start,end):
    truncated_x = []
    truncated_y = []

    for time, value in zip(x,y):
        if start <= time <= end:
            truncated_x.append(time)
            truncated_y.append(value)

    return truncated_x, truncated_y

def spliner(x,y,smooth):
    sp = csaps.CubicSmoothingSpline(x,y,smooth = smooth)
    fit = sp(x)
    return fit

def ratio(y1,y2):
    rat = []
    for i,j in zip(y1,y2):
        rat.append((i/j) - 1)
    return rat

def hilbert_envelope(y):
    analytic = hilbert(y)
    envelope = np.abs(analytic)
    return envelope


def fft(x, y, f_min=0.01, f_max=10.0, num_points=2000):
    x = np.array(x)
    y = np.array(y)

    f = np.linspace(f_min, f_max, num_points)
    angular_freqs = 2 * np.pi * f

    power = lombscargle(x, y, angular_freqs, normalize=True)
    peak_idx = np.argmax(power)
    peak_freq = f[peak_idx]
    print('FFT Frequency: ' + str(peak_freq))

    return f.tolist(), power.tolist()

def trim_and_sum(x, *y_lists):
    """
    Trims x and all y_lists to the shortest y length, then sums the y_lists element-wise.

    Parameters:
        x (list): The reference x list.
        *y_lists (lists): Variable number of y lists to be summed.

    Returns:
        x_trimmed (list): Trimmed x list.
        y_sum (list): Element-wise sum of trimmed y lists.
    """
    # Find the shortest length among all y lists and x
    min_len = min(len(x), *(len(y) for y in y_lists))

    # Trim x and all y lists to min_len
    x_trimmed = x[:min_len]
    y_trimmed = [y[:min_len] for y in y_lists]

    # Element-wise sum
    y_sum = [sum(values) for values in zip(*y_trimmed)]

    return x_trimmed, y_sum


def smooth_bump_region(x, y, x_min, x_max, window_length=11, polyorder=3):
    x = np.array(x)
    y = np.array(y)

    y_smoothed = y.copy()
    bump_mask = (x >= x_min) & (x <= x_max)

    # Apply smoothing only to the bump region
    y_smoothed[bump_mask] = savgol_filter(y[bump_mask], window_length, polyorder)

    return y_smoothed

def flatten_bump(x, y, x_min, x_max):
    x = np.array(x)
    y = np.array(y)
    y_fixed = y.copy()

    bump_mask = (x >= x_min) & (x <= x_max)
    indices = np.where(bump_mask)[0]

    if len(indices) == 0:
        raise ValueError("No data points in the bump region.")

    left_idx = indices[0] - 1
    right_idx = indices[-1] + 1

    if left_idx < 0 or right_idx >= len(x):
        raise ValueError("Bump too close to edge for interpolation.")

    # Interpolate across the bump
    y_fixed[indices] = np.interp(x[indices], [x[left_idx], x[right_idx]], [y[left_idx], y[right_idx]])

    return y_fixed

def suppress_bump_gently(x, y, x_min, x_max, window_length=7, polyorder=2, blend=0.8):
    """
    Suppresses a bump by blending the original signal with a smoothed version in a specific region.
    The 'blend' factor controls how much to favor the smoothed version (0.0 = original, 1.0 = full smooth).
    """
    x = np.array(x)
    y = np.array(y)
    y_fixed = y.copy()

    bump_mask = (x >= x_min) & (x <= x_max)
    indices = np.where(bump_mask)[0]

    if len(indices) < window_length:
        window_length = len(indices) if len(indices) % 2 == 1 else len(indices) - 1

    if window_length < 3:
        return y_fixed  # too short to smooth

    y_smooth = savgol_filter(y[indices], window_length, polyorder)

    # Blend the smoothed and original
    y_fixed[indices] = (1 - blend) * y[indices] + blend * y_smooth

    return y_fixed

def scale_bump_region(x, y, x_min, x_max, factor):
    """
    Multiplies the values of y in the range [x_min, x_max] by a constant factor.

    Parameters:
        x (array-like): The x values.
        y (array-like): The y values.
        x_min (float): Start of bump region.
        x_max (float): End of bump region.
        factor (float): Multiplier to apply to the bump region.

    Returns:
        y_scaled (np.ndarray): The modified y values.
    """
    x = np.array(x)
    y = np.array(y)
    y_scaled = y.copy()

    bump_mask = (x >= x_min) & (x <= x_max)
    y_scaled[bump_mask] *= factor

    return y_scaled

def exponential_fit(x, y):
    def exponential(x, a, k):
        return a * np.exp(-x / k)
    x = np.array(x)
    y = np.array(y)
    # Perform the curve fit
    popt, pcov = curve_fit(exponential, x, y, p0=[0.05, 10], maxfev=50000)

    # Generate fitted y-values
    y_fit = exponential(x, *popt)

    # Extract standard deviations (sqrt of diagonal of covariance matrix)
    perr = np.sqrt(np.sqrt(np.diag(pcov)))
    a_std = perr[0]
    k_std = perr[1]

    print(f"Time Constant (k): {popt[1]:.5f} ± {k_std:.5f}")
    print(f"Pre-Factor (a): {popt[0]:.5f} ± {a_std:.5f}")

    return y_fit

def custom_fit(x,y):
    def custom(x,a,b,c,d,f,g):
        return a*np.exp(-x/b) + c*np.exp(-x/d) + f*g*x

    popt_exponential, pcov_exponential = scipy.optimize.curve_fit(custom, x , y, p0 = [1,1,1,1,1,1], maxfev = 100000)

    y_val = []
    for i in x:
        value = custom(i, popt_exponential[0], popt_exponential[1],popt_exponential[2], popt_exponential[3],
                       popt_exponential[4], popt_exponential[5])
        y_val.append(value)

    return y_val


def extract_column(dataframe, n):
    column = dataframe.iloc[: , n-1]
    column = list(column)
    return column

def truncate_to_peak(y):
    listt = []
    for i in range(y.index(max(y)), len(y)):
        listt.append(y[i])
    del listt[0]
    return listt

def time_converter(y,int):
    listt = []
    for i in range(0,len(y)):
        listt.append(i * int)
    return listt


def normalize(y):
    if not y:
        return []

    peak = max(y)
    if peak == 0:
        return [0] * len(y)

    return [y / peak for y in y]

def log_bin(x, y, num_bins):
    """
    Logarithmically bin x and average y values in each bin.

    Parameters:
    x (list or array-like): x values (must be positive).
    y (list or array-like): y values.
    num_bins (int): Number of logarithmic bins.

    Returns:
    tuple: (binned_x, binned_y) where both are numpy arrays.
    """
    del x[0]
    del y[0]
    x = np.array(x)
    y = np.array(y)

    # Ensure all x values are positive
    if np.any(x <= 0):
        raise ValueError("All x values must be positive for logarithmic binning.")

    # Create logarithmically spaced bin edges
    bins = np.logspace(np.log10(min(x)), np.log10(max(x)), num_bins + 1)
    binned_x = []
    binned_y = []

    for i in range(num_bins):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        if np.any(mask):
            binned_x.append(np.mean(x[mask]))
            binned_y.append(np.mean(y[mask]))

    return np.array(binned_x), np.array(binned_y)


def accumulated_count(x, y):
    x = np.array(x)
    y = np.array(y)

    # Compute cumulative sum of photon counts
    cumulative_counts = np.cumsum(y)

    return x, cumulative_counts
