import numpy as np
from scipy.linalg import expm
import os

# === Constants and Parameters ===
bohr_magneton = 1.78e-2  # 1 / Gauss·ns
dZFS = 1.6564
eZFS = -0.11872
theta_deg = 31
theta = np.deg2rad(theta_deg)

# === NEW: angle sweeps in degrees ===
mag_theta_degs = [39.6]   # <-- edit as you like
mag_phi_degs = [0]      # <-- edit as you like

nn = np.array([0.1, 0.2, 0.3])
int_str = 8e-8
override = 'none'  # options: 'AA', 'AB', 'BB', or 'none'

time_end = 10
twoD_tau = 0.250
time_res = 90
sim_res = 200

# === NEW: long-time fusion probability after disentanglement ===
P_infty = 1.0 / 9.0  # spin-statistical fusion probability after disentanglement

# === NEW: sweep parameters ===
field_strengths_gauss = [0,300,2000,8000]  # <-- edit as you like
decoh_times_ns = [1000000000000000000]  # <-- edit as you like (ns); can include 0

# === Helper Functions ===
def rotation_matrix_z(angle_rad):
    return np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad),  np.cos(angle_rad), 0],
        [0, 0, 1]
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
        [mag_diag, -magZMini,  magYMini],
        [magZMini, mag_diag,  -magXMini],
        [-magYMini, magXMini,  mag_diag]
    ])

def interaction_matrix(nn):
    nnX, nnY, nnZ = nn
    Z = np.array([[0, 1-3*nnZ**2, 3*nnY*nnZ],
                  [-1+3*nnZ**2, 0, -3*nnX*nnZ],
                  [-3*nnY*nnZ, 3*nnX*nnZ, 0]])
    Y = np.array([[0, 3*nnY*nnZ, 1-3*nnY**2],
                  [-3*nnY*nnZ, 0, 3*nnX*nnY],
                  [-1+3*nnY**2, -3*nnX*nnY, 0]])
    X = np.array([[0, -3*nnX*nnZ, 3*nnX*nnY],
                  [3*nnX*nnZ, 0, 1-3*nnX**2],
                  [-3*nnX*nnY, -1+3*nnX**2, 0]])
    return int_str * np.block([
        [np.zeros((3,3)), Z, Y],
        [-Z, np.zeros((3,3)), X],
        [-Y, -X, np.zeros((3,3))]
    ])

# === Hamiltonians: field-independent parts ===
rotA = rotation_matrix_z(theta)
rotB = rotation_matrix_z(-theta)

zero_mini = 2*np.pi * np.array([
    [-2*dZFS/3, 0, 0],
    [0, dZFS/3 - eZFS, 0],
    [0, 0, dZFS/3 + eZFS]
], dtype=complex)

zero_AA = rotA.T @ zero_mini @ rotA
zero_BB = rotB.T @ zero_mini @ rotB

hAA_base = np.kron(zero_AA, np.eye(3)) + np.kron(np.eye(3), zero_AA)
hBB_base = np.kron(zero_BB, np.eye(3)) + np.kron(np.eye(3), zero_BB)
hAB_base = np.kron(zero_AA, np.eye(3)) + np.kron(np.eye(3), zero_BB)

hAA_base = hAA_base.astype(complex)
hBB_base = hBB_base.astype(complex)
hAB_base = hAB_base.astype(complex)

# Dipolar interaction (field-independent)
intMat = interaction_matrix(nn).astype(complex)

# Time axis (same as yours)
x = np.linspace(0, time_end, time_end * time_res)

# Singlet (same as yours)
singlet = (1/np.sqrt(3)) * np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=complex)

# Output folder
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop/ch8_data")

# === Main sweep ===
for mag_theta_deg in mag_theta_degs:
    for mag_phi_deg in mag_phi_degs:

        mag_theta = np.deg2rad(mag_theta_deg)
        mag_phi = np.deg2rad(mag_phi_deg)

        run_name = f"phase_{mag_theta_deg:03g}_{mag_phi_deg:03g}"

        for mag_field_strength in field_strengths_gauss:

            # Zeeman depends on B and angle
            zeeman = find_h_zee(mag_theta, mag_phi, mag_field_strength, bohr_magneton).astype(complex)

            # Full Hamiltonians for this B and angle
            hAA = hAA_base + intMat + zeeman
            hBB = hBB_base + intMat + zeeman
            hAB = hAB_base + intMat + zeeman
            hams = [hAA, hAB, hBB]

            for tau_decoh in decoh_times_ns:

                result = np.zeros((sim_res, len(x)))

                for i, t in enumerate(x):
                    print(
                        f"theta={mag_theta_deg} deg | phi={mag_phi_deg} deg | "
                        f"B={mag_field_strength} G | tau_decoh={tau_decoh} ns | t={t:.6f} ns"
                    )

                    for j in range(sim_res):
                        if override in ['AA', 'AB', 'BB']:
                            idx = {'AA': 0, 'AB': 1, 'BB': 2}[override]
                            psi = expm(-1j * hams[idx] * t) @ singlet
                        else:
                            curr_time = 0.0
                            psi = singlet.copy()
                            tp_con = np.random.choice([0, 1, 2])

                            while True:
                                next_hop = np.random.exponential(twoD_tau)

                                if curr_time + next_hop > t:
                                    next_hop = t - curr_time
                                    psi = expm(-1j * hams[tp_con] * next_hop) @ psi
                                    break

                                psi = expm(-1j * hams[tp_con] * next_hop) @ psi
                                curr_time += next_hop

                                if tp_con == 1:
                                    tp_con = np.random.choice([0, 2])
                                else:
                                    tp_con = 1

                        # Project onto singlet
                        proj = np.vdot(singlet, psi)
                        P_spin = np.abs(proj)**2  # coherent spin-0 probability

                        # Apply decoherence towards 1/9
                        if tau_decoh > 0:
                            decoh_factor = np.exp(-t / tau_decoh)
                            P_eff = P_infty + (P_spin - P_infty) * decoh_factor
                        else:
                            # tau_decoh = 0 means "instantly randomized"
                            P_eff = P_infty

                        result[j, i] = P_eff

                # Average trace for this angle, B, and tau_decoh
                avg_raw = np.mean(result, axis=0)

                # Save to TXT
                filename = f"{run_name}_{mag_field_strength}gauss_{tau_decoh}dec.txt"
                file_path = os.path.join(desktop_path, filename)

                data_to_save = np.column_stack((x, avg_raw))
                np.savetxt(
                    file_path,
                    data_to_save,
                    header="Time (ns)\tAverage Decohered Spin-0 Probability",
                    fmt="%.6f",
                    delimiter="\t"
                )

                print(f"Saved: {file_path}")