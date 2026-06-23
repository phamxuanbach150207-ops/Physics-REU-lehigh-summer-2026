"""
Boosts_Quench_curves.py
Computes and plots the BQ ratio (field-on / zero-field integrated PL yield)
for every column in the angle-sweep dataset.
Column 1 = zero-field reference.
Columns 2–20 = field-on at different angles → BQ ratios plotted vs. column index.
"""

import numpy as np
import matplotlib.pyplot as plt
from Data_Toolbox import (
    get_data, extract_column, truncate_to_peak,
    normalize, time_converter, integrated_yield_in_window
)

# Parameters 
DATA_PATH = r"F:\Physics REU Lehigh 2026\Data\june 8\tetracene_lf_angle_sweep.xlsx"
DT        = 0.016
T1, T2    = 0.0, 10.0
N_COLS    = 20
ZF_COL    = 1

data = get_data(DATA_PATH)

# Zero-field reference 
y_zf = normalize(truncate_to_peak(extract_column(data, ZF_COL)))
t_zf = time_converter(y_zf, DT)
zf_integral = integrated_yield_in_window(t_zf, y_zf, T1, T2)
print(f"Zero-field integral: {zf_integral:.4f}")

# Loop over field-on columns
col_indices = []
bq_ratios   = []

for col in range(2, N_COLS + 1):
    y = normalize(truncate_to_peak(extract_column(data, col)))
    t = time_converter(y, DT)
    integral = integrated_yield_in_window(t, y, T1, T2)
    ratio    = integral / zf_integral
    col_indices.append(col - 1)   # memory block index
    bq_ratios.append(ratio)
    print(f"  col {col:>2d} (block {col-1:>2d}): BQ = {ratio:.4f}")

# Plot
fig, ax = plt.subplots(figsize=(10, 5))

ax.scatter(col_indices, bq_ratios, color='steelblue', zorder=3)
ax.plot(col_indices, bq_ratios, color='steelblue', alpha=0.5)
ax.axhline(y=1, color='black', linewidth=1, linestyle='--', label='zero-field baseline')

ax.set_xlabel('Memory block index (angle step)')
ax.set_ylabel('Integrated PL ratio (field / zero-field)')
ax.set_title(f'BQ curve — integration window [{T1}, {T2}] ns')
ax.set_ylim(0.8, 1.4)
ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()
