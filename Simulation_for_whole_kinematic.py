import numpy as np
from numba import njit, prange
from tqdm import tqdm
import multiprocessing as mp
import time
import os

# ============================================================
# USER SWEEPS
# ============================================================

PFUS_TXT_PATHS = [
    "/Users/zacharyrex/Desktop/ch8_data/phase_39.6_000_300gauss_1000000000000000000dec.txt"
]

HOP_1D_NS_LIST = [0.003]

TAU_HOP2D_NS_LIST = [0.200]

FISS_PS_LIST = [10]

TAU_DECOH_NS_LIST = [10000000000]

# ============================================================
# FUSION PROBABILITY MODE
# ============================================================
# Options:
#   PFUS_MODE = "table"
#       Uses your P_fus(t) txt table while coherent,
#       and P_INFTY = 1/9 after decoherence.
#
#   PFUS_MODE = "constant"
#       Ignores P_fus(t), ignores decoherence, and always uses
#       CONSTANT_PFUS as the fusion probability after an attempt.
# ============================================================

PFUS_MODE = "table"        # "table" or "constant"
CONSTANT_PFUS = 1.0 / 9.0  # only used if PFUS_MODE = "constant"

# ============================================================
# GLOBAL CONSTANTS
# ============================================================

t_end = np.float32(10.0)

attempt_prob = np.float32(0.3)

P_INFTY = np.float32(1.0 / 9.0)

TAU_EMIT_NS = np.float32(0.99)

N_TOTAL = 25_000_000
BATCH_SIZE = 10_000

MAX_RESETS = 256
MAX_DECISIONS = 4096
MAX_ATTEMPTS = 4096

DESKTOP = os.path.join(os.path.expanduser("~"), "Desktop")
OUT_DIR = os.path.join(DESKTOP, "ch8_data")
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# WORKER GLOBALS
# ============================================================

T_TAB = None
P_TAB = None

# ============================================================
# HELPERS
# ============================================================

def base_tag_from_txt(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def hop_tag_from_ns(tau_ns: float) -> str:
    ps = int(round(float(tau_ns) * 1000.0))
    return f"{ps:04d}"

def fiss_tag_from_ps(fiss_ps: int) -> str:
    return f"{int(fiss_ps):03d}"

def decoh_tag_from_ns(tau_ns: float) -> str:
    x = float(tau_ns)
    if abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    return f"{x:.6g}".replace(".", "p")

def pfus_mode_tag() -> str:
    if PFUS_MODE == "constant":
        return f"CONSTpfus_{CONSTANT_PFUS:.6g}".replace(".", "p")
    elif PFUS_MODE == "table":
        return "TABLEpfus"
    else:
        raise ValueError("PFUS_MODE must be either 'table' or 'constant'")

# ============================================================
# NUMBA HELPERS
# ============================================================

@njit
def exp_sample(tau):
    return -tau * np.log(np.random.rand())

@njit
def pfus_lookup_nearest(age, t_tab, p_tab):
    n = t_tab.size

    if n == 0:
        return np.float32(0.0)

    if age <= t_tab[0]:
        return p_tab[0]

    if age >= t_tab[n - 1]:
        return p_tab[n - 1]

    lo = 0
    hi = n - 1

    while hi - lo > 1:
        mid = (lo + hi) // 2
        if t_tab[mid] < age:
            lo = mid
        else:
            hi = mid

    if age - t_tab[lo] <= t_tab[hi] - age:
        return p_tab[lo]
    else:
        return p_tab[hi]

# ============================================================
# NUMBA SIMULATION
# ============================================================

@njit(parallel=True)
def simulate_batch_full_analysis(
    n,
    t_end,
    hop_1d,
    hop_2d,
    attempt_prob,
    total_rate,
    p_emit,
    max_resets,
    max_decisions,
    max_attempts,
    t_tab,
    p_tab,
    tau_decoh_ns,
    p_infty,
    pfus_mode_constant,
    constant_pfus
):
    photon_times = np.full(n, -1.0, dtype=np.float32)
    last_reset_before_emit = np.full(n, -1.0, dtype=np.float32)

    emitted = np.zeros(n, dtype=np.int8)

    # -1 = no photon
    #  1 = photon emitted from coherent/correlated fusion
    #  0 = photon emitted from decohered fusion, p_fus = 1/9
    #
    # In constant-p_fus mode, this is set to 1 by convention
    # because coherent/decohered status is not used to determine p_fus.
    photon_coherent = np.full(n, -1, dtype=np.int8)

    encounter_counts = np.zeros(n, dtype=np.int32)
    attempt_counts = np.zeros(n, dtype=np.int32)
    fusion_success_counts = np.zeros(n, dtype=np.int32)
    refission_counts = np.zeros(n, dtype=np.int32)

    reset_counts = np.zeros(n, dtype=np.int32)

    decision_truncated = np.zeros(n, dtype=np.int8)
    attempt_truncated = np.zeros(n, dtype=np.int8)
    reset_truncated = np.zeros(n, dtype=np.int8)

    decision_time_buf = np.full((n, max_decisions), -1.0, dtype=np.float32)
    decision_age_buf = np.full((n, max_decisions), -1.0, dtype=np.float32)

    attempt_time_buf = np.full((n, max_attempts), -1.0, dtype=np.float32)
    attempt_age_buf = np.full((n, max_attempts), -1.0, dtype=np.float32)
    attempt_pfus_buf = np.full((n, max_attempts), -1.0, dtype=np.float32)
    attempt_success_buf = np.zeros((n, max_attempts), dtype=np.int8)

    reset_time_buf = np.full((n, max_resets), -1.0, dtype=np.float32)

    for run in prange(n):

        posA = np.zeros(2, dtype=np.int32)
        posB = np.zeros(2, dtype=np.int32)

        t = np.float32(0.0)
        last_fission_time = np.float32(0.0)

        t_decoh = exp_sample(tau_decoh_ns)

        n_enc = 0
        n_att = 0
        n_fus = 0
        n_refiss = 0
        n_reset = 1

        reset_time_buf[run, 0] = np.float32(0.0)

        Ax = exp_sample(hop_1d)
        Ay = exp_sample(hop_2d)
        Bx = exp_sample(hop_1d)
        By = exp_sample(hop_2d)

        while t < t_end:

            t_next = Ax
            if Ay < t_next:
                t_next = Ay
            if Bx < t_next:
                t_next = Bx
            if By < t_next:
                t_next = By

            t = t_next

            if t >= t_end:
                break

            if t == Ax:
                posA[0] += -1 if np.random.rand() < 0.5 else 1
                Ax = t + exp_sample(hop_1d)

            elif t == Ay:
                posA[1] += -1 if np.random.rand() < 0.5 else 1
                Ay = t + exp_sample(hop_2d)

            elif t == Bx:
                posB[0] += -1 if np.random.rand() < 0.5 else 1
                Bx = t + exp_sample(hop_1d)

            else:
                posB[1] += -1 if np.random.rand() < 0.5 else 1
                By = t + exp_sample(hop_2d)

            if posA[0] == posB[0] and posA[1] == posB[1]:

                age = t - last_fission_time

                if n_enc < max_decisions:
                    decision_time_buf[run, n_enc] = t
                    decision_age_buf[run, n_enc] = age
                else:
                    decision_truncated[run] = 1

                n_enc += 1

                if np.random.rand() >= attempt_prob:
                    continue

                # ====================================================
                # FUSION PROBABILITY CHOICE
                # ====================================================
                # Mode 1: constant
                #   Always use constant_pfus.
                #
                # Mode 2: table
                #   If coherent: use P_fus(age) from table.
                #   If decohered: use p_infty = 1/9.
                # ====================================================

                if pfus_mode_constant == 1:
                    fusion_p = constant_pfus
                    this_attempt_coherent = np.int8(1)
                else:
                    if age < t_decoh:
                        fusion_p = pfus_lookup_nearest(age, t_tab, p_tab)
                        this_attempt_coherent = np.int8(1)
                    else:
                        fusion_p = p_infty
                        this_attempt_coherent = np.int8(0)

                if fusion_p < 0.0:
                    fusion_p = np.float32(0.0)
                elif fusion_p > 1.0:
                    fusion_p = np.float32(1.0)

                if n_att < max_attempts:
                    attempt_time_buf[run, n_att] = t
                    attempt_age_buf[run, n_att] = age
                    attempt_pfus_buf[run, n_att] = fusion_p
                else:
                    attempt_truncated[run] = 1

                n_att += 1

                if np.random.rand() < fusion_p:

                    n_fus += 1

                    if n_att <= max_attempts:
                        attempt_success_buf[run, n_att - 1] = 1

                    t += exp_sample(np.float32(1.0) / total_rate)

                    if t >= t_end:
                        break

                    if np.random.rand() < p_emit:

                        photon_times[run] = t
                        last_reset_before_emit[run] = last_fission_time
                        emitted[run] = 1
                        photon_coherent[run] = this_attempt_coherent

                        break

                    else:
                        n_refiss += 1

                        posA[0] = 0
                        posA[1] = 0
                        posB[0] = 0
                        posB[1] = 0

                        last_fission_time = t

                        if n_reset < max_resets:
                            reset_time_buf[run, n_reset] = t
                        else:
                            reset_truncated[run] = 1

                        n_reset += 1

                        t_decoh = exp_sample(tau_decoh_ns)

                        Ax = t + exp_sample(hop_1d)
                        Ay = t + exp_sample(hop_2d)
                        Bx = t + exp_sample(hop_1d)
                        By = t + exp_sample(hop_2d)

        encounter_counts[run] = n_enc
        attempt_counts[run] = n_att
        fusion_success_counts[run] = n_fus
        refission_counts[run] = n_refiss
        reset_counts[run] = n_reset

    return (
        photon_times,
        last_reset_before_emit,
        emitted,
        photon_coherent,
        encounter_counts,
        attempt_counts,
        fusion_success_counts,
        refission_counts,
        reset_counts,
        decision_truncated,
        attempt_truncated,
        reset_truncated,
        decision_time_buf,
        decision_age_buf,
        attempt_time_buf,
        attempt_age_buf,
        attempt_pfus_buf,
        attempt_success_buf,
        reset_time_buf
    )

# ============================================================
# WORKER INIT
# ============================================================

def _init_worker(pfus_txt_path):
    global T_TAB, P_TAB

    data = np.loadtxt(pfus_txt_path)

    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"P_fus txt must have >=2 columns. Got shape {data.shape}")

    t = data[:, 0].astype(np.float32)
    p = data[:, 1].astype(np.float32)

    order = np.argsort(t)

    T_TAB = t[order]
    P_TAB = p[order]

# ============================================================
# FLATTEN HELPERS
# ============================================================

def flatten_buffer(counts, buf, max_count):
    recorded_counts = np.minimum(counts, np.int32(max_count)).astype(np.int64)

    offsets = np.zeros(len(counts), dtype=np.int64)
    if len(counts) > 1:
        offsets[1:] = np.cumsum(recorded_counts[:-1])

    total_flat = int(np.sum(recorded_counts))
    flat = np.empty(total_flat, dtype=buf.dtype)

    ptr = 0
    for i in range(len(counts)):
        c = int(recorded_counts[i])
        if c > 0:
            flat[ptr:ptr + c] = buf[i, :c]
            ptr += c

    return offsets, flat

# ============================================================
# WORKER
# ============================================================

def _worker(args):
    from numba import set_num_threads
    set_num_threads(1)

    global T_TAB, P_TAB

    (
        batch_n,
        t_end,
        hop_1d,
        hop_2d,
        attempt_prob,
        total_rate,
        p_emit,
        max_resets,
        max_decisions,
        max_attempts,
        tau_decoh_ns,
        p_infty,
        pfus_mode_constant,
        constant_pfus
    ) = args

    results = simulate_batch_full_analysis(
        batch_n,
        t_end,
        hop_1d,
        hop_2d,
        attempt_prob,
        total_rate,
        p_emit,
        max_resets,
        max_decisions,
        max_attempts,
        T_TAB,
        P_TAB,
        np.float32(tau_decoh_ns),
        np.float32(p_infty),
        np.int8(pfus_mode_constant),
        np.float32(constant_pfus)
    )

    (
        photon_times,
        last_reset_before_emit,
        emitted,
        photon_coherent,
        encounter_counts,
        attempt_counts,
        fusion_success_counts,
        refission_counts,
        reset_counts,
        decision_truncated,
        attempt_truncated,
        reset_truncated,
        decision_time_buf,
        decision_age_buf,
        attempt_time_buf,
        attempt_age_buf,
        attempt_pfus_buf,
        attempt_success_buf,
        reset_time_buf
    ) = results

    decision_offsets, decision_times_flat = flatten_buffer(
        encounter_counts,
        decision_time_buf,
        max_decisions
    )

    _, decision_ages_flat = flatten_buffer(
        encounter_counts,
        decision_age_buf,
        max_decisions
    )

    attempt_offsets, attempt_times_flat = flatten_buffer(
        attempt_counts,
        attempt_time_buf,
        max_attempts
    )

    _, attempt_ages_flat = flatten_buffer(
        attempt_counts,
        attempt_age_buf,
        max_attempts
    )

    _, attempt_pfus_flat = flatten_buffer(
        attempt_counts,
        attempt_pfus_buf,
        max_attempts
    )

    _, attempt_success_flat = flatten_buffer(
        attempt_counts,
        attempt_success_buf,
        max_attempts
    )

    reset_offsets, reset_times_flat = flatten_buffer(
        reset_counts,
        reset_time_buf,
        max_resets
    )

    return (
        photon_times,
        last_reset_before_emit,
        emitted,
        photon_coherent,
        encounter_counts,
        attempt_counts,
        fusion_success_counts,
        refission_counts,
        reset_counts,
        decision_truncated,
        attempt_truncated,
        reset_truncated,
        decision_offsets,
        decision_times_flat,
        decision_ages_flat,
        attempt_offsets,
        attempt_times_flat,
        attempt_ages_flat,
        attempt_pfus_flat,
        attempt_success_flat,
        reset_offsets,
        reset_times_flat
    )

# ============================================================
# MULTIPROCESS RUN
# ============================================================

def run_multiprocess(
    pool,
    n_total,
    batch_size,
    t_end,
    hop_1d,
    hop_2d,
    attempt_prob,
    total_rate,
    p_emit,
    max_resets,
    max_decisions,
    max_attempts,
    tau_decoh_ns,
    p_infty,
    pfus_mode_constant,
    constant_pfus,
    desc=""
):
    n_full = n_total // batch_size
    rem = n_total - n_full * batch_size

    tasks = []

    for _ in range(n_full):
        tasks.append((
            batch_size,
            t_end,
            hop_1d,
            hop_2d,
            attempt_prob,
            total_rate,
            p_emit,
            max_resets,
            max_decisions,
            max_attempts,
            tau_decoh_ns,
            p_infty,
            pfus_mode_constant,
            constant_pfus
        ))

    if rem > 0:
        tasks.append((
            rem,
            t_end,
            hop_1d,
            hop_2d,
            attempt_prob,
            total_rate,
            p_emit,
            max_resets,
            max_decisions,
            max_attempts,
            tau_decoh_ns,
            p_infty,
            pfus_mode_constant,
            constant_pfus
        ))

    all_photon_times = []
    all_last_reset = []
    all_emitted = []
    all_photon_coherent = []

    all_encounters = []
    all_attempts = []
    all_fusions = []
    all_refissions = []
    all_resets = []

    all_decision_trunc = []
    all_attempt_trunc = []
    all_reset_trunc = []

    all_decision_times = []
    all_decision_ages = []
    all_attempt_times = []
    all_attempt_ages = []
    all_attempt_pfus = []
    all_attempt_success = []
    all_reset_times = []

    for result in tqdm(
        pool.imap_unordered(_worker, tasks, chunksize=1),
        total=len(tasks),
        desc=desc
    ):
        (
            photon_times,
            last_reset_before_emit,
            emitted,
            photon_coherent,
            encounter_counts,
            attempt_counts,
            fusion_success_counts,
            refission_counts,
            reset_counts,
            decision_truncated,
            attempt_truncated,
            reset_truncated,
            decision_offsets_local,
            decision_times_flat,
            decision_ages_flat,
            attempt_offsets_local,
            attempt_times_flat,
            attempt_ages_flat,
            attempt_pfus_flat,
            attempt_success_flat,
            reset_offsets_local,
            reset_times_flat
        ) = result

        all_photon_times.append(photon_times)
        all_last_reset.append(last_reset_before_emit)
        all_emitted.append(emitted)
        all_photon_coherent.append(photon_coherent)

        all_encounters.append(encounter_counts)
        all_attempts.append(attempt_counts)
        all_fusions.append(fusion_success_counts)
        all_refissions.append(refission_counts)
        all_resets.append(reset_counts)

        all_decision_trunc.append(decision_truncated)
        all_attempt_trunc.append(attempt_truncated)
        all_reset_trunc.append(reset_truncated)

        all_decision_times.append(decision_times_flat)
        all_decision_ages.append(decision_ages_flat)

        all_attempt_times.append(attempt_times_flat)
        all_attempt_ages.append(attempt_ages_flat)
        all_attempt_pfus.append(attempt_pfus_flat)
        all_attempt_success.append(attempt_success_flat)

        all_reset_times.append(reset_times_flat)

    photon_times_all = np.concatenate(all_photon_times)
    last_reset_before_emit_all = np.concatenate(all_last_reset)
    emitted_all = np.concatenate(all_emitted)
    photon_coherent_all = np.concatenate(all_photon_coherent)

    encounter_counts_all = np.concatenate(all_encounters)
    attempt_counts_all = np.concatenate(all_attempts)
    fusion_success_counts_all = np.concatenate(all_fusions)
    refission_counts_all = np.concatenate(all_refissions)
    reset_counts_all = np.concatenate(all_resets)

    decision_truncated_all = np.concatenate(all_decision_trunc)
    attempt_truncated_all = np.concatenate(all_attempt_trunc)
    reset_truncated_all = np.concatenate(all_reset_trunc)

    decision_times_flat_all = np.concatenate(all_decision_times)
    decision_ages_flat_all = np.concatenate(all_decision_ages)

    attempt_times_flat_all = np.concatenate(all_attempt_times)
    attempt_ages_flat_all = np.concatenate(all_attempt_ages)
    attempt_pfus_flat_all = np.concatenate(all_attempt_pfus)
    attempt_success_flat_all = np.concatenate(all_attempt_success)

    reset_times_flat_all = np.concatenate(all_reset_times)

    decision_recorded_counts = np.minimum(
        encounter_counts_all,
        np.int32(max_decisions)
    ).astype(np.int64)

    attempt_recorded_counts = np.minimum(
        attempt_counts_all,
        np.int32(max_attempts)
    ).astype(np.int64)

    reset_recorded_counts = np.minimum(
        reset_counts_all,
        np.int32(max_resets)
    ).astype(np.int64)

    decision_offsets_all = np.zeros(n_total, dtype=np.int64)
    attempt_offsets_all = np.zeros(n_total, dtype=np.int64)
    reset_offsets_all = np.zeros(n_total, dtype=np.int64)

    if n_total > 1:
        decision_offsets_all[1:] = np.cumsum(decision_recorded_counts[:-1])
        attempt_offsets_all[1:] = np.cumsum(attempt_recorded_counts[:-1])
        reset_offsets_all[1:] = np.cumsum(reset_recorded_counts[:-1])

    return (
        photon_times_all,
        last_reset_before_emit_all,
        emitted_all,
        photon_coherent_all,
        encounter_counts_all,
        attempt_counts_all,
        fusion_success_counts_all,
        refission_counts_all,
        reset_counts_all,
        decision_truncated_all,
        attempt_truncated_all,
        reset_truncated_all,
        decision_offsets_all,
        decision_times_flat_all,
        decision_ages_flat_all,
        attempt_offsets_all,
        attempt_times_flat_all,
        attempt_ages_flat_all,
        attempt_pfus_flat_all,
        attempt_success_flat_all,
        reset_offsets_all,
        reset_times_flat_all
    )

# ============================================================
# WARMUP
# ============================================================

def warmup_compile(pfus_txt_path):
    data = np.loadtxt(pfus_txt_path)

    t = data[:, 0].astype(np.float32)
    p = data[:, 1].astype(np.float32)

    order = np.argsort(t)
    t = t[order]
    p = p[order]

    tau_fission_ns = np.float32(FISS_PS_LIST[0] / 1000.0)

    rate_emit = np.float32(1.0) / TAU_EMIT_NS
    rate_fiss = np.float32(1.0) / tau_fission_ns

    total_rate = rate_emit + rate_fiss
    p_emit = rate_emit / total_rate

    pfus_mode_constant = np.int8(1 if PFUS_MODE == "constant" else 0)

    _ = simulate_batch_full_analysis(
        10,
        t_end,
        np.float32(HOP_1D_NS_LIST[0]),
        np.float32(TAU_HOP2D_NS_LIST[0]),
        attempt_prob,
        total_rate,
        p_emit,
        MAX_RESETS,
        MAX_DECISIONS,
        MAX_ATTEMPTS,
        t,
        p,
        np.float32(TAU_DECOH_NS_LIST[0]),
        P_INFTY,
        pfus_mode_constant,
        np.float32(CONSTANT_PFUS)
    )

# ============================================================
# MAIN
# ============================================================

def main():
    overall_t0 = time.time()

    if PFUS_MODE not in ["table", "constant"]:
        raise ValueError("PFUS_MODE must be either 'table' or 'constant'")

    if CONSTANT_PFUS < 0.0 or CONSTANT_PFUS > 1.0:
        raise ValueError("CONSTANT_PFUS must be between 0 and 1")

    for pth in PFUS_TXT_PATHS:
        if not os.path.exists(pth):
            raise FileNotFoundError(f"P_fus txt not found: {pth}")

    n_proc = max(1, mp.cpu_count() - 1)

    pfus_mode_constant = np.int8(1 if PFUS_MODE == "constant" else 0)
    mode_tag = pfus_mode_tag()

    print(f"Using {n_proc} processes")
    print(f"Output folder: {OUT_DIR}")
    print(f"N_TOTAL={N_TOTAL:,}")
    print(f"BATCH_SIZE={BATCH_SIZE:,}")
    print(f"t_end={float(t_end)} ns")
    print(f"attempt_prob={float(attempt_prob)}")
    print(f"TAU_EMIT_NS={float(TAU_EMIT_NS)} ns")
    print(f"PFUS_MODE={PFUS_MODE}")
    print(f"CONSTANT_PFUS={float(CONSTANT_PFUS)}")
    print(f"MAX_DECISIONS={MAX_DECISIONS}")
    print(f"MAX_ATTEMPTS={MAX_ATTEMPTS}")
    print(f"MAX_RESETS={MAX_RESETS}")
    print()

    print("Warming up Numba compilation...")
    warmup_compile(PFUS_TXT_PATHS[0])
    print("Warmup complete.\n")

    ctx = mp.get_context("spawn")

    for pfus_path in PFUS_TXT_PATHS:
        jtag = base_tag_from_txt(pfus_path)

        print(f"\n=== TABLE: {jtag} ===")

        if PFUS_MODE == "table":
            print(f"Using P_fus table from: {pfus_path}\n")
        else:
            print(f"Using constant P_fus = {float(CONSTANT_PFUS)}")
            print("P_fus table is only loaded to keep the same multiprocessing structure.\n")

        with ctx.Pool(
            processes=n_proc,
            maxtasksperchild=50,
            initializer=_init_worker,
            initargs=(pfus_path,)
        ) as pool:

            for hop_1d_val in HOP_1D_NS_LIST:
                hop_1d = np.float32(hop_1d_val)
                hop1d_tag = hop_tag_from_ns(hop_1d_val)

                for tau_hop2d in TAU_HOP2D_NS_LIST:
                    hop_2d = np.float32(tau_hop2d)
                    hop2d_tag = hop_tag_from_ns(tau_hop2d)

                    for fiss_ps in FISS_PS_LIST:
                        fiss_ps = int(fiss_ps)
                        fiss_tag = fiss_tag_from_ps(fiss_ps)

                        tau_fission_ns = np.float32(fiss_ps / 1000.0)

                        rate_emit = np.float32(1.0) / TAU_EMIT_NS
                        rate_fiss = np.float32(1.0) / tau_fission_ns

                        total_rate = rate_emit + rate_fiss
                        p_emit = rate_emit / total_rate

                        for tau_decoh in TAU_DECOH_NS_LIST:
                            tau_decoh = float(tau_decoh)
                            decoh_tag = decoh_tag_from_ns(tau_decoh)

                            filename = (
                                f"{jtag}_{mode_tag}_1d{hop1d_tag}hop_2d{hop2d_tag}hop_"
                                f"{fiss_tag}fiss_{decoh_tag}decohere_FULL_ANALYSIS.npz"
                            )

                            save_path = os.path.join(OUT_DIR, filename)

                            desc = (
                                f"{jtag} | {mode_tag} | 1d={hop1d_tag}ps "
                                f"2d={hop2d_tag}ps fiss={fiss_ps}ps full"
                            )

                            t0 = time.time()

                            results = run_multiprocess(
                                pool=pool,
                                n_total=N_TOTAL,
                                batch_size=BATCH_SIZE,
                                t_end=t_end,
                                hop_1d=hop_1d,
                                hop_2d=hop_2d,
                                attempt_prob=attempt_prob,
                                total_rate=total_rate,
                                p_emit=p_emit,
                                max_resets=MAX_RESETS,
                                max_decisions=MAX_DECISIONS,
                                max_attempts=MAX_ATTEMPTS,
                                tau_decoh_ns=np.float32(tau_decoh),
                                p_infty=P_INFTY,
                                pfus_mode_constant=pfus_mode_constant,
                                constant_pfus=np.float32(CONSTANT_PFUS),
                                desc=desc
                            )

                            (
                                photon_times_all,
                                last_reset_before_emit_all,
                                emitted_all,
                                photon_coherent_all,
                                encounter_counts_all,
                                attempt_counts_all,
                                fusion_success_counts_all,
                                refission_counts_all,
                                reset_counts_all,
                                decision_truncated_all,
                                attempt_truncated_all,
                                reset_truncated_all,
                                decision_offsets_all,
                                decision_times_flat_all,
                                decision_ages_flat_all,
                                attempt_offsets_all,
                                attempt_times_flat_all,
                                attempt_ages_flat_all,
                                attempt_pfus_flat_all,
                                attempt_success_flat_all,
                                reset_offsets_all,
                                reset_times_flat_all
                            ) = results

                            np.savez(
                                save_path,

                                photon_times_all=photon_times_all,
                                last_reset_before_emit_all=last_reset_before_emit_all,
                                emitted_all=emitted_all,

                                photon_coherent_all=photon_coherent_all,

                                encounter_counts_all=encounter_counts_all,
                                attempt_counts_all=attempt_counts_all,
                                fusion_success_counts_all=fusion_success_counts_all,
                                refission_counts_all=refission_counts_all,
                                reset_counts_all=reset_counts_all,

                                decision_truncated_all=decision_truncated_all,
                                attempt_truncated_all=attempt_truncated_all,
                                reset_truncated_all=reset_truncated_all,

                                decision_offsets_all=decision_offsets_all,
                                decision_times_flat_all=decision_times_flat_all,
                                decision_ages_flat_all=decision_ages_flat_all,

                                attempt_offsets_all=attempt_offsets_all,
                                attempt_times_flat_all=attempt_times_flat_all,
                                attempt_ages_flat_all=attempt_ages_flat_all,
                                attempt_pfus_flat_all=attempt_pfus_flat_all,
                                attempt_success_flat_all=attempt_success_flat_all,

                                reset_offsets_all=reset_offsets_all,
                                reset_times_flat_all=reset_times_flat_all,

                                meta=np.array([
                                    float(t_end),
                                    float(hop_1d),
                                    float(hop_2d),
                                    float(tau_fission_ns),
                                    float(TAU_EMIT_NS),
                                    float(attempt_prob),
                                    float(p_emit),
                                    float(tau_decoh),
                                    float(P_INFTY),
                                    float(CONSTANT_PFUS),
                                    int(pfus_mode_constant),
                                    int(MAX_DECISIONS),
                                    int(MAX_ATTEMPTS),
                                    int(MAX_RESETS),
                                    int(N_TOTAL)
                                ], dtype=np.float64),

                                meta_labels=np.array([
                                    "t_end_ns",
                                    "hop_1d_ns",
                                    "hop_2d_ns",
                                    "tau_fission_ns",
                                    "TAU_EMIT_NS",
                                    "attempt_prob",
                                    "p_emit_given_singlet",
                                    "tau_decoh_ns",
                                    "P_INFTY",
                                    "CONSTANT_PFUS",
                                    "PFUS_MODE_CONSTANT",
                                    "MAX_DECISIONS",
                                    "MAX_ATTEMPTS",
                                    "MAX_RESETS",
                                    "N_TOTAL"
                                ], dtype=object),

                                photon_coherent_labels=np.array([
                                    "-1 = no photon emitted",
                                    "0 = photon emitted after decoherence, p_fus = 1/9",
                                    "1 = photon emitted while coherent, p_fus from table OR constant-p_fus mode"
                                ], dtype=object),

                                pfus_mode=np.array([PFUS_MODE], dtype=object),
                                pfus_table=np.array([pfus_path], dtype=object)
                            )

                            n_emit = int(np.sum(emitted_all))
                            n_emit_coh = int(np.sum(photon_coherent_all == 1))
                            n_emit_decoh = int(np.sum(photon_coherent_all == 0))

                            n_dec_trunc = int(np.sum(decision_truncated_all))
                            n_att_trunc = int(np.sum(attempt_truncated_all))
                            n_rst_trunc = int(np.sum(reset_truncated_all))

                            print(
                                f"\n✅ Saved -> {save_path}\n"
                                f"   PFUS_MODE = {PFUS_MODE}\n"
                                f"   CONSTANT_PFUS = {float(CONSTANT_PFUS):.6g}\n"
                                f"   emitted photons = {n_emit:,} / {N_TOTAL:,}\n"
                                f"   coherent-label emitted photons = {n_emit_coh:,}\n"
                                f"   decohered-label emitted photons = {n_emit_decoh:,}\n"
                                f"   photon probability = {n_emit / N_TOTAL:.6g}\n"
                                f"   mean encounters / trajectory = {np.mean(encounter_counts_all):.6g}\n"
                                f"   mean attempts / trajectory = {np.mean(attempt_counts_all):.6g}\n"
                                f"   mean successful fusions / trajectory = {np.mean(fusion_success_counts_all):.6g}\n"
                                f"   mean refissions / trajectory = {np.mean(refission_counts_all):.6g}\n"
                                f"   decision truncation = {n_dec_trunc:,}\n"
                                f"   attempt truncation = {n_att_trunc:,}\n"
                                f"   reset truncation = {n_rst_trunc:,}\n"
                                f"   p_emit_given_singlet = {float(p_emit):.6g}\n"
                                f"⏱️ Run time: {time.time() - t0:.2f} s\n"
                            )

    print(f"\n🏁 ALL DONE. Total wall time: {time.time() - overall_t0:.2f} s\n")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()