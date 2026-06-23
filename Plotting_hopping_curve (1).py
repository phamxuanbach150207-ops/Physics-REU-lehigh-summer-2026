"""
Plotting_hopping_curve.py
-------------------------
Plots all angle-sweep TCSPC curves from the dataset on a single log-scale figure.
Column 1 = zero-field reference (plotted in black).
Columns 2–20 = field-on curves at different angles (plotted in color).
"""

import numpy as np
import matplotlib.pyplot as plt
from Data_Toolbox import (
    get_data, extract_column, truncate_to_peak,
    normalize, time_converter, integrated_yield_in_window
)

# ── Parameters ────────────────────────────────────────────────────────────────
DATA_PATH = r"F:\Physics REU Lehigh 2026\Data\june 8\tetracene_lf_angle_sweep.xlsx"
DT        = 0.016       # ns per channel
T1, T2    = 0.0, 10.0  # integration window (ns)
N_COLS    = 20
ZF_COL    = 1           # 1-indexed zero-field column

# ── Load ──────────────────────────────────────────────────────────────────────
data = get_data(DATA_PATH)

# ── Zero-field reference ──────────────────────────────────────────────────────
y_zf = normalize(truncate_to_peak(extract_column(data, ZF_COL)))
t_zf = time_converter(y_zf, DT)
zf_integral = integrated_yield_in_window(t_zf, y_zf, T1, T2)
print(f"Zero-field integral: {zf_integral:.4f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
cmap = plt.cm.viridis

ax.plot(t_zf, y_zf, color='black', linewidth=1.5, label='Zero field (col 1)', zorder=5)

for col in range(2, N_COLS + 1):
    y = normalize(truncate_to_peak(extract_column(data, col)))
    t = time_converter(y, DT)
    integral = integrated_yield_in_window(t, y, T1, T2)
    color = cmap((col - 2) / (N_COLS - 2))
    ax.plot(t, y, color=color, linewidth=0.8, alpha=0.7, label=f'Col {col} | BQ={integral/zf_integral:.3f}')
    print(f"  col {col:>2d}: BQ = {integral/zf_integral:.4f}")

ax.set_xlim(0, 15)
ax.set_yscale('log')
ax.set_xlabel('t [ns]')
ax.set_ylabel('PL(t) (normalized)')
ax.set_title('Angle sweep — all curves')
ax.grid(alpha=0.3)
ax.legend(fontsize=6, ncol=2, loc='upper right')

plt.tight_layout()
plt.show()
