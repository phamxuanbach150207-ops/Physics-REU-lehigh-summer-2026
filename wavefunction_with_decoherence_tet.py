import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import os

# === Constants and Parameters ===
bohr_magneton = 1.78e-2  # 1 / Gauss·ns

# === Tetracene ZFS parameters from Yarmus 1972 ===
dZFS = 1.5590
eZFS = -0.1559

# === Tetracene herringbone angle ===
theta_deg = 26
theta = np.deg2rad(theta_deg)

# === Single Target Orientation & Field (No Sweeps) ===
mag_theta_deg = 0
mag_phi_deg = 0
mag_field_strength = 0  # 0 Gauss to capture clean zero-field beats

mag_theta = np.deg2rad(mag_theta_deg)
mag_phi = np.deg2rad(mag_phi_deg)

# === Interaction & Hopping Parameters ===
nn = np.array([0.1, 0.2, 0.3])
int_str = 8e-8
override = 'none'  # options: 'AA', 'AB', 'BB', or 'none'

time_end = 5
twoD_tau = 0.005
time_res = 90
sim_res = 200

# === Decoherence parameters ===
tau_decoh = 1000000000000000000  # Long lifetime
P_infty = 1.0 / 9.0


# ============================================================
# Helper Functions
# ============================================================

def rotation_matrix_x(angle_rad):
    return np.array([
        [1, 0, 0],
        [0, np.cos(angle_rad), -np.sin(angle_rad)],
        [0, np.sin(angle_rad), np.cos(angle_rad)]
    ])


def find_h_zee(theta_rad, phi_rad, B, bM):
    magX = B * np.sin(theta_rad) * np.cos(phi_rad)
    magY = B * np.sin(theta_rad) * np.sin(phi_rad)
    magZ = B * np.cos(theta_rad)

    mag_diag = np.array([
        [0, -magZ, magY],
        [magZ, 0, -magX],
        [-magY, magX, 0]
    ])

    magXMini = magX * np.eye(3)
    magYMini = magY * np.eye(3)
    magZMini = magZ * np.eye(3)

    return 1j * bM * np.block([
        [mag_diag, -magZMini, magYMini],
        [magZMini, mag_diag, -magXMini],
        [-magYMini, magXMini, mag_diag]
    ])


def interaction_matrix(nn):
    nnX, nnY, nnZ = nn
    Z = np.array([[0, 1 - 3 * nnZ ** 2, 3 * nnY * nnZ],
                  [-1 + 3 * nnZ ** 2, 0, -3 * nnX * nnZ],
                  [-3 * nnY * nnZ, 3 * nnX * nnZ, 0]])
    Y = np.array([[0, 3 * nnY * nnZ, 1 - 3 * nnY ** 2],
                  [-3 * nnY * nnZ, 0, 3 * nnX * nnY],
                  [-1 + 3 * nnY ** 2, -3 * nnX * nnY, 0]])
    X = np.array([[0, -3 * nnX * nnZ, 3 * nnX * nnY],
                  [3 * nnX * nnZ, 0, 1 - 3 * nnX ** 2],
                  [-3 * nnX * nnY, -1 + 3 * nnX ** 2, 0]])
    return int_str * np.block([
        [np.zeros((3, 3)), Z, Y],
        [-Z, np.zeros((3, 3)), X],
        [-Y, -X, np.zeros((3, 3))]
    ])


# ============================================================
# Hamiltonian Assembly
# ============================================================

rotA = rotation_matrix_x(theta)
rotB = rotation_matrix_x(-theta)

# Reorder zero_mini to standard Cartesian (X, Y, Z)
zero_mini = 2 * np.pi * np.array([
    [dZFS/3 - eZFS, 0, 0],       # Row 1: X-axis
    [0, dZFS/3 + eZFS, 0],       # Row 2: Y-axis
    [0, 0, -2*dZFS/3]            # Row 3: Z-axis
], dtype=complex)

zero_AA = rotA.T @ zero_mini @ rotA
zero_BB = rotB.T @ zero_mini @ rotB

hAA_base = np.kron(zero_AA, np.eye(3)) + np.kron(np.eye(3), zero_AA)
hBB_base = np.kron(zero_BB, np.eye(3)) + np.kron(np.eye(3), zero_BB)
hAB_base = np.kron(zero_AA, np.eye(3)) + np.kron(np.eye(3), zero_BB)

intMat = 1j * interaction_matrix(nn).astype(complex)
zeeman = find_h_zee(mag_theta, mag_phi, mag_field_strength, bohr_magneton).astype(complex)

hams = [
    hAA_base + intMat + zeeman,
    hAB_base + intMat + zeeman,
    hBB_base + intMat + zeeman
]

# ============================================================
# Simulation Grid & Execution
# ============================================================

num_points = int(time_end * time_res)
x = np.linspace(0, time_end, num_points)
singlet = (1 / np.sqrt(3)) * np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=complex)
result = np.zeros((sim_res, len(x)))

print(f"Starting simulation for Tetracene at B = {mag_field_strength} G...")

for j in range(sim_res):
    psi = singlet.copy()
    curr_time = 0.0

    if override in ['AA', 'AB', 'BB']:
        idx = {'AA': 0, 'AB': 1, 'BB': 2}[override]
        for i, t in enumerate(x):
            psi_t = expm(-1j * hams[idx] * t) @ singlet
            proj = np.vdot(singlet, psi_t)
            P_spin = np.abs(proj) ** 2
            result[j, i] = P_infty + (P_spin - P_infty) * np.exp(-t / tau_decoh) if tau_decoh > 0 else P_infty
    else:
        tp_con = np.random.choice([0, 1, 2])
        next_hop = np.random.exponential(twoD_tau)

        for i, t in enumerate(x):
            while curr_time + next_hop < t:
                dt = next_hop
                psi = expm(-1j * hams[tp_con] * dt) @ psi
                curr_time += dt
                tp_con = np.random.choice([0, 2]) if tp_con == 1 else 1
                next_hop = np.random.exponential(twoD_tau)

            dt_rem = t - curr_time
            psi_at_t = expm(-1j * hams[tp_con] * dt_rem) @ psi

            proj = np.vdot(singlet, psi_at_t)
            P_spin = np.abs(proj) ** 2

            if tau_decoh > 0:
                result[j, i] = P_infty + (P_spin - P_infty) * np.exp(-t / tau_decoh)
            else:
                result[j, i] = P_infty

# Calculate the average trace across all stochastic paths
avg_raw = np.mean(result, axis=0)

# ============================================================
# File Export
# ============================================================

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop/tetracene_p_fuse")
os.makedirs(desktop_path, exist_ok=True)

filename = f"phase_{mag_theta_deg:03g}_{mag_phi_deg:03g}_{mag_field_strength}gauss_{tau_decoh}dec.txt"
file_path = os.path.join(desktop_path, filename)

data_to_save = np.column_stack((x, avg_raw))
np.savetxt(
    file_path,
    data_to_save,
    header="Time (ns)\tAverage Decohered Spin-0 Probability",
    fmt="%.6f",
    delimiter="\t"
)
print(f"Data file saved successfully to: {file_path}")

# ============================================================
# Plotting
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(x, avg_raw, color='dodgerblue', lw=2, label=r'$\langle P_{\mathrm{fusion}}(t) \rangle$')
plt.axhline(P_infty, linestyle='--', color='tomato', alpha=0.7, label='Statistical Limit (1/9)')
plt.xlabel('Time [ns]', fontsize=12)
plt.ylabel('Fusion Probability', fontsize=12)
plt.title(f'Tetracene Fusion Dynamics (B = {mag_field_strength} G)', fontsize=13, fontweight='bold')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()