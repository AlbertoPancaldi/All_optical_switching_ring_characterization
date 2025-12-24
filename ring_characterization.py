"""
Created on Jan 2026
@author: albertopancaldi

Combined ring + MZI characterization script.
CH1 carries the ring resonances, CH2 carries the MZI fringes used to calibrate
the time-to-wavelength mapping. Designed for Spyder with cell markers.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

#%% ---------------- User inputs ----------------

# Data file location
FOLDER_PATH = r'/Users/albertopancaldi/Documents/Python/PIC/Internship/ring_mzi_heat/'    # Folder where your files are stored
CSV_FILE = "ring2_left_wgbelow_mzi_curr900_1545_1555_sweep_5.csv"                                                # CSV file name

# Wavelength sweep bounds 
LAMBDA_START = 1545.0
LAMBDA_END   = 1555.0

# Sweep velocity in nm/s
SWEEP_VELOCITY_NM_S = 5.0

# Constants
n_g_mzi = 1.467                 # group index of the optical fiber used to build the MZI at 1550 nm
n_g_ring = 2.028                # group index of the InGaP waveguide at 1550 nm
C0 = 299792458.0                # m/s


#%% -------- Read CSV raw --------

def read_csv(path=FOLDER_PATH, file=CSV_FILE):
    """Read scope CSV, detect the 'Second'/'Volt' header, return t_s, ch1_v, ch2_v."""
    if not os.path.isdir(path):
        raise NotADirectoryError(f"Folder not found: {path}")
    file_path = os.path.join(path, file)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file: {file_path}")

    header_idx = None
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            low = line.lower()
            if "second" in low and "volt" in low:
                header_idx = i
                break
    if header_idx is None:
        raise RuntimeError("Couldn't find a header line containing both 'Second' and 'Volt'.")

    df_raw = pd.read_csv(
        file_path,
        header=header_idx,
        sep=None,
        engine="python",
        dtype=str
    )

    norm = {c: c.strip().lower() for c in df_raw.columns}
    df_raw.rename(columns=norm, inplace=True)

    time_candidates = [c for c in df_raw.columns if "second" in c or c in ("time", "t", "seconds", "sec", "s")]
    volt_candidates = [c for c in df_raw.columns if "volt" in c or c in ("v", "voltage")]
    if not time_candidates or len(volt_candidates) < 2:
        raise RuntimeError("Expected one time column and two voltage columns (CH1/CH2).")

    t_col = time_candidates[0]
    v1_col, v2_col = volt_candidates[:2]
    df = df_raw[[t_col, v1_col, v2_col]].copy()
    df.columns = ["t_s", "ch1_v", "ch2_v"]

    for col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    df = df.dropna(subset=df.columns).reset_index(drop=True)
    return df


raw_df = read_csv(FOLDER_PATH, CSV_FILE)

# First visualization of raw data
plt.figure(figsize=(10, 4))
plt.plot(raw_df["t_s"], raw_df["ch1_v"], label="CH1 - raw", alpha=0.6)
plt.plot(raw_df["t_s"], raw_df["ch2_v"], label="CH2 - raw", alpha=0.6)
plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")
plt.grid(True)
plt.legend()
plt.title("Raw data from CSV")
plt.tight_layout()
plt.show()


#%% -------- Offset removal and downsampling --------

AUTOMATIC_OFFSET_REMOVAL = False             # set to False to skip the automatic offset removal read from csv header
KNOWN_OFFSETS = {"ch1": 0.2, "ch2": 1}     # set known offsets here if AUTOMATIC_OFFSET_REMOVAL is False
                                             # if no offsets are known, leave as {0.0, 0.0}

USE_DOWNSAMPLING = False        # set to True to enable downsampling
DOWNSAMPLE_FACTOR = 10          # number of samples to skip when downsampling


def vertical_offset_removal(df, path, file, max_lines=40):
    """Subtract CH1/CH2 offsets parsed from the CSV header."""
    file_path = os.path.join(path, file)
    
    offsets = {"ch1": 0.0, "ch2": 0.0}
    pattern = re.compile(r"CH(?P<ch>[12]):(?P<val>[+-]?\d*\.?\d+(?:[Ee][+-]?\d+)?)")
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for _, line in zip(range(max_lines), f):
            if line.lower().startswith("vertical offset"):
                for match in pattern.finditer(line):
                    ch = match.group("ch")
                    offsets[f"ch{ch}"] = float(match.group("val"))
                break

    df_corr = df.copy()
    df_corr["ch1_v"] = df_corr["ch1_v"] - offsets.get("ch1", 0.0)
    df_corr["ch2_v"] = df_corr["ch2_v"] - offsets.get("ch2", 0.0)
    return df_corr, offsets

def downsample_dataframe(df, factor=10):
    """Take every N-th sample across all columns."""
    factor = max(1, int(factor))
    return df.iloc[::factor].reset_index(drop=True)

if AUTOMATIC_OFFSET_REMOVAL:
    df, header_offsets = vertical_offset_removal(raw_df, FOLDER_PATH, CSV_FILE)
    print(f"\nOffsets removed: CH1={header_offsets.get('ch1', 0.0):.4e}, CH2={header_offsets.get('ch2', 0.0):.4e}")
else:
    df = raw_df.copy()
    df["ch1_v"] = df["ch1_v"] + KNOWN_OFFSETS.get("ch1", 0.0)
    df["ch2_v"] = df["ch2_v"] + KNOWN_OFFSETS.get("ch2", 0.0)
    if KNOWN_OFFSETS.get("ch1", 0.0) != 0.0 or KNOWN_OFFSETS.get("ch2", 0.0) != 0.0:
        print(f"\nKnown offsets removed: CH1={KNOWN_OFFSETS.get('ch1', 0.0):.4e}, CH2={KNOWN_OFFSETS.get('ch2', 0.0):.4e}")

if USE_DOWNSAMPLING:
    df = downsample_dataframe(df, DOWNSAMPLE_FACTOR)
    print(f"Downsampled by {DOWNSAMPLE_FACTOR} → {len(df)} rows")

if AUTOMATIC_OFFSET_REMOVAL or KNOWN_OFFSETS.get("ch1", 0.0) != 0.0 or KNOWN_OFFSETS.get("ch2", 0.0) != 0.0:
    # Visualization after offset removal
    plt.figure(figsize=(10, 4))
    plt.plot(df["t_s"], df["ch1_v"], label="CH1", alpha=0.6)
    plt.plot(df["t_s"], df["ch2_v"], label="CH2", alpha=0.6)
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.grid(True)
    plt.legend()
    plt.title("Data after offset removal")
    plt.tight_layout()
    plt.show()


#%% -------- Optional channel swap (CH1 ↔ CH2) --------
# Set swapped_channels = True if, in this CSV, CH1 and CH2 are swapped with respect to the expected mapping:
#   expected: CH1 = ring, CH2 = MZI

swapped_channels = True        # set to True only for files with swapped channels

if swapped_channels:
    # Swap the voltage columns so that downstream code still uses
    # df["ch1_v"] for the ring and df["ch2_v"] for the MZI.
    df = df.rename(columns={"ch1_v": "tmp_ch1"})
    df = df.rename(columns={"ch2_v": "ch1_v"})
    df = df.rename(columns={"tmp_ch1": "ch2_v"})
    print("\n[Channel swap] CH1 and CH2 have been swapped: CH1→ring, CH2→MZI\n")


#%% -------- Auto-detect active time window from MZI --------
# Slope-based detector: assumes flat regions have slope ~0 (± slope_tol) and detects when slope deviates.

SELECT_WINDOW_MANUALLY = False          # set True to manually limit time span
MANUAL_T_MIN = -0.25                   # requested minimum time [s]
MANUAL_T_MAX = 1.8                     # requested maximum time [s]

USE_AUTOMATIC_WINDOW_DETECTION = True   # set to False to skip automatic window detection
SLOPE_REF = 0.0                         # reference slope value
SLOPE_TOL = 0.55                         # ± tolerance around slope=0 to detect the MZI active region
DETECTION_WINDOW_LENGTH = 0.02          # window length as a fraction of the trace length
OVERLAP_FRACTION = 0.6                  # overlap between consecutive windows as a fraction of window length
WAVELENGTH_REGION_TOL = 0.05            # tolerance for wavelength region check (as a fraction of total Δλ)


# Optional manual pre-windowing before automatic detection
if SELECT_WINDOW_MANUALLY:
    full_t_min = float(df["t_s"].min())
    full_t_max = float(df["t_s"].max())
    t_min_req = float(MANUAL_T_MIN)
    t_max_req = float(MANUAL_T_MAX)
    t_min = max(t_min_req, full_t_min)
    t_max = min(t_max_req, full_t_max)

    if t_min >= t_max:
        print(
            f"\n[Manual pre-window] Requested [{t_min_req:.4f}, {t_max_req:.4f}] s "
            f"is invalid or outside data range [{full_t_min:.4f}, {full_t_max:.4f}] s. "
            "Skipping manual cut and using full time trace."
        )
    else:
        mask = (df["t_s"] >= t_min) & (df["t_s"] <= t_max)
        df = df[mask].reset_index(drop=True)
        print("\n[Manual pre-window] Enabled.")
        print(f"Requested window: [{t_min_req:.4f}, {t_max_req:.4f}] s")
        print(f"Applied window:   [{df['t_s'].min():.4f}, {df['t_s'].max():.4f}] s "
              f"(data range was [{full_t_min:.4f}, {full_t_max:.4f}] s)")
        plt.figure(figsize=(10, 4))
        plt.plot(df['t_s'], df['ch1_v'], label='CH1 (ring)', alpha=0.6)
        plt.plot(df['t_s'], df['ch2_v'], label='CH2 (MZI)', alpha=0.6)
        plt.xlabel("Time [s]")
        plt.ylabel("Voltage [V]")
        plt.grid(True)
        plt.legend()
        plt.title("Data after manual pre-windowing")
        plt.tight_layout()
        plt.show()


def find_active_window(times, voltage, window_fraction):
    times = np.asarray(times).flatten()
    voltage = np.asarray(voltage).flatten()

    n = len(times)
    window_n = max(int(window_fraction * n), 1)                  # window_n: number of samples per window
    step_n = max(1, int(round(window_n * OVERLAP_FRACTION)))     # step_n: shift between successive windows

    slopes = []
    centers = []

    # Left-to-right scan: find first deviation from slope ~0
    found_left = False
    t_start_idx = 0
    i0 = 0
    while i0 < n:
        i1 = min(i0 + window_n - 1, n - 1)
        if i1 <= i0:
            break
        # Compute average slope in this window via linear regression (v vs t)
        t_w = times[i0:i1 + 1]
        v_w = voltage[i0:i1 + 1]
        if len(t_w) < 2 or not np.isfinite(t_w).all():
            slope = np.nan
        else:
            t_mean = np.mean(t_w)
            v_mean = np.mean(v_w)
            num = np.sum((t_w - t_mean) * (v_w - v_mean))
            den = np.sum((t_w - t_mean) ** 2)
            slope = num / den if den > 0 else np.nan
        slopes.append(slope)
        centers.append((i0 + i1) // 2)
        if np.isfinite(slope) and abs(slope - SLOPE_REF) > SLOPE_TOL and not found_left:
            t_start_idx = i0
            found_left = True
            break
        i0 += step_n
        if i1 == n - 1:
            break
    if not found_left:
        print("[find_active_window] Left scan found no deviation; using start=0.")

    # Right-to-left scan: find first deviation from slope ~0
    found_right = False
    t_end_idx = n - 1
    i1 = n - 1
    while i1 >= 0:
        i0 = max(i1 - window_n + 1, 0)
        if i0 >= i1:
            break
        # Compute average slope in this window via linear regression (v vs t)
        t_w = times[i0:i1 + 1]
        v_w = voltage[i0:i1 + 1]
        if len(t_w) < 2 or not np.isfinite(t_w).all():
            slope = np.nan
        else:
            t_mean = np.mean(t_w)
            v_mean = np.mean(v_w)
            num = np.sum((t_w - t_mean) * (v_w - v_mean))
            den = np.sum((t_w - t_mean) ** 2)
            slope = num / den if den > 0 else np.nan
        slopes.append(slope)
        centers.append((i0 + i1) // 2)
        if np.isfinite(slope) and abs(slope - SLOPE_REF) > SLOPE_TOL and not found_right:
            t_end_idx = i1
            found_right = True
            break
        if i0 == 0:
            break
        i1 -= step_n
    if not found_right:
        print("[find_active_window] Right scan found no deviation; using end=n-1.")

    # Build a debug array with slopes at window centers; rest NaN
    debug_arr = np.full(n, np.nan)
    for c, s in zip(centers, slopes):
        if 0 <= c < n:
            debug_arr[c] = s

    return times[t_start_idx], times[t_end_idx], (t_start_idx, t_end_idx), debug_arr

if USE_AUTOMATIC_WINDOW_DETECTION:
    t_start, t_end, (i_start, i_end), debug_arr = find_active_window(
        df["t_s"].to_numpy(),
        df["ch2_v"].to_numpy(),
        window_fraction=DETECTION_WINDOW_LENGTH,
    )

    times = df["t_s"].to_numpy()
    mask = ~np.isnan(debug_arr)

    plt.figure(figsize=(10, 4))
    plt.plot(times[mask], debug_arr[mask], ".-", label="Window slopes")
    plt.axhline(SLOPE_REF, color="k", linestyle="--", label="slope reference")
    plt.axhline(-SLOPE_TOL, color="r", linestyle=":", label="slope tolerance")
    plt.axhline(SLOPE_TOL, color="r", linestyle=":")
    plt.axvline(t_start, color="g", linestyle="--", label="t_start")
    plt.axvline(t_end, color="m", linestyle="--", label="t_end")
    plt.xlabel("Time [s]")
    plt.ylabel("Slope")
    plt.grid(True)
    plt.legend()
    plt.title("MZI (CH2) slope-based active window detection")
    plt.tight_layout()
    plt.show()

    print(f"\nAuto-selected time window: [{t_start:.4f}, {t_end:.4f}]s (samples {i_start}-{i_end})")
    
else:
    t_start = df["t_s"].min()
    t_end = df["t_s"].max()


if USE_AUTOMATIC_WINDOW_DETECTION:
    df = df[(df["t_s"] >= t_start) & (df["t_s"] <= t_end)].reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No samples left after automatic windowing.")

    plt.figure(figsize=(10, 4))
    plt.plot(df["t_s"], df["ch1_v"], label="CH1 (ring)", alpha=0.6)
    plt.plot(df["t_s"], df["ch2_v"], label="CH2 (MZI)", alpha=0.6)
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.grid(True)
    plt.legend()
    plt.title("Data after automatic windowing")
    plt.tight_layout()
    plt.show()

# Sanity check: does the time span match the expected wavelength span given SWEEP_VELOCITY_NM_S?
t0, t1 = float(t_start), float(t_end)
delta_from_v = abs(SWEEP_VELOCITY_NM_S * (t1 - t0))
delta_target = abs(LAMBDA_END - LAMBDA_START) 
if abs(delta_from_v - delta_target) > WAVELENGTH_REGION_TOL * abs(delta_target):
    CUT_SIGNAL = True
    print(f"\nWARNING: sweep speed * time window implies Δλ ≈ {delta_from_v:.3f} nm, "
        f"but (LAMBDA_END - LAMBDA_START) is {delta_target:.3f} nm. "
        "\nCheck time window, sweep velocity (nm/s) or wavelength bounds.")
else:
    CUT_SIGNAL = False
    print(f"\nThe considered time window corresponds to Δλ≈{delta_from_v:.3f} nm, "
        f"matching (LAMBDA_END - LAMBDA_START) = {delta_target:.3f} nm.")

#%% -------- MZI FSR extraction on CH2 --------

# Optional: use pre-characterized MZI instead of deriving FSR_nm / ΔL from the sweep
MZI_FSR_NM_KNOWN = False        # set True if you know MZI FSR in nm from a previous calibration
MZI_FSR_NM_VALUE = 2.6          # if MZI_FSR_NM_KNOWN is True, set the known FSR value here [nm]

DELTA_L_KNOWN = False           # set True if you know ΔL of the MZI arms from design/measurement
DELTA_L_VALUE = 6.4e-04         # if DELTA_L_KNOWN is True, set the known ΔL value here [m]


def estimate_mzi_fsr(times, voltage):
    """Fit a sine to CH2 to extract the MZI FSR in time."""
    df_local = pd.DataFrame({"t_s": times, "v": voltage}).sort_values("t_s", ignore_index=True)
    t = df_local["t_s"].to_numpy()
    y = df_local["v"].to_numpy()

    y0 = y - np.mean(y)
    N = len(t)
    if N < 10:
        raise RuntimeError("Not enough samples for MZI fit.")

    dt = float(np.median(np.diff(t)))
    f = np.fft.rfftfreq(N, d=dt)
    Y = np.fft.rfft(y0)
    if len(f) < 2:
        raise RuntimeError("Not enough frequency bins for FFT.")
    k1 = 1 + np.argmax(np.abs(Y[1:]))
    f0_guess = max(f[k1], 1.0 / (t[-1] - t[0] + 1e-12))    # Guard against unrealistically low freq guesses (e.g. DC-dominated FFT)
    w0_guess = 2 * np.pi * f0_guess

    A_guess = 0.5 * (np.max(y) - np.min(y))
    B_guess = float(np.mean(y))
    phi_guess = 0.0

    def sfunc(t, A, w, phi, B):
        return A * np.sin(w * t + phi) + B

    lb = (-np.inf, 1e-6, -2 * np.pi, -np.inf)
    ub = (np.inf, np.inf, 2 * np.pi, np.inf)

    p0 = (A_guess, w0_guess, phi_guess, B_guess)
    popt, _ = curve_fit(sfunc, t, y, p0=p0, bounds=(lb, ub), maxfev=20000)
    A, w0, phi, B = popt

    # FSR from fitted period (time domain)
    T = 2 * np.pi / w0
    fsr_time_s = float(T)  # always use fitted period in time

    lambda_center_nm = 0.5 * (LAMBDA_START + LAMBDA_END)
    lam_c_m = lambda_center_nm * 1e-9

    if MZI_FSR_NM_KNOWN and DELTA_L_KNOWN:
        print(
            "\nWARNING: Both MZI_FSR_NM_KNOWN and DELTA_L_KNOWN are True; "
            "using MZI_FSR_NM_KNOWN and ignoring DELTA_L_KNOWN."
        )

    # --- Case 1: user provides known MZI FSR in nm ---
    if MZI_FSR_NM_KNOWN:
        fsr_nm_known = float(MZI_FSR_NM_VALUE)
        fsr_nm_fit = fsr_time_s * SWEEP_VELOCITY_NM_S
        fsr_nm = fsr_nm_known
        fsr_hz = C0 * (fsr_nm * 1e-9) / (lam_c_m**2)
        delta_L_m = C0 / (n_g_mzi * fsr_hz)
        if fsr_nm_fit > 0:
            rel_diff = abs(fsr_nm_fit - fsr_nm_known) / fsr_nm_fit
            if rel_diff > 0.05:
                print(
                    f"\nWARNING: MZI_FSR_NM_KNOWN = {fsr_nm_known:.6e} nm, "
                    f"but fit + sweep gives ≈ {fsr_nm_fit:.6e} nm "
                    f"(rel. diff ≈ {100*rel_diff:.1f}%)."
                )

    # --- Case 2: user provides known ΔL (arm length difference) ---
    elif DELTA_L_KNOWN:
        delta_L_known = float(DELTA_L_VALUE)
        fsr_hz = C0 / (n_g_mzi * delta_L_known)
        fsr_nm = (fsr_hz * lam_c_m**2) / C0
        delta_L_m = delta_L_known
        fsr_nm_fit = fsr_time_s * SWEEP_VELOCITY_NM_S
        if fsr_nm_fit > 0:
            rel_diff = abs(fsr_nm_fit - fsr_nm) / fsr_nm_fit
            if rel_diff > 0.05:
                print(
                    f"\nWARNING: ΔL_KNOWN implies FSR≈{fsr_nm:.6e} nm, "
                    f"but fit + sweep gives ≈ {fsr_nm_fit:.6e} nm "
                    f"(rel. diff ≈ {100*rel_diff:.1f}%)."
                )

    # --- Default: derive FSR_nm and ΔL from fitted period + sweep velocity ---
    else:
        fsr_nm = fsr_time_s * SWEEP_VELOCITY_NM_S
        fsr_hz = C0 * (fsr_nm * 1e-9) / (lam_c_m**2)
        delta_L_m = C0 / (n_g_mzi * fsr_hz)

    phase_for_max = (np.pi / 2.0) if A >= 0 else (-np.pi / 2.0)
    tmin, tmax = t[0], t[-1]
    k_start = int(np.floor((w0 * tmin + phi - phase_for_max) / (2 * np.pi))) - 1
    k_end = int(np.ceil((w0 * tmax + phi - phase_for_max) / (2 * np.pi))) + 1

    t_peaks_fit = []
    for k in range(k_start, k_end + 1):
        tp = (phase_for_max - phi + 2 * np.pi * k) / w0
        if tmin <= tp <= tmax:
            t_peaks_fit.append(tp)
    t_peaks_fit = np.array(sorted(t_peaks_fit))
    peaks_idx = np.searchsorted(t, t_peaks_fit)
    peaks_idx = np.clip(peaks_idx, 0, len(t) - 1)

    # --- Check sweep linearity via MZI fringe spacing in time ---
    if len(t_peaks_fit) >= 3:
        dt_peaks = np.diff(t_peaks_fit)
        dt_mean = float(np.mean(dt_peaks))
        dt_std = float(np.std(dt_peaks, ddof=1))
        rel_std = dt_std / dt_mean if dt_mean > 0 else np.inf
        if rel_std > 0.05:
            print(
                f"\nWARNING: MZI fringe spacing in time is not very uniform "
                f"(mean Δt≈{dt_mean:.3e} s, std≈{dt_std:.3e} s, rel.std≈{100*rel_std:.1f}%)."
                "\nThis suggests nonlinearity in the sweep or timing jitter; "
                "FSR/ΔL extraction may be less reliable."
            )

    results = {
        "fsr_time_s": fsr_time_s,
        "fsr_nm": fsr_nm,
        "fsr_hz": fsr_hz,
        "delta_L_m": delta_L_m,
        "fit_params": (A, w0, phi, B),
        "t_sorted": t,
        "y_sorted": y,
    }
    return results, peaks_idx


def plot_with_peaks(df_local, peaks_idx, fit_params, label):
    """Plot data, fitted sine, and peak markers."""
    plt.figure(figsize=(10, 4))
    plt.plot(df_local["t_s"], df_local["v"], alpha=0.4, label=label)
    A, w0, phi, B = fit_params
    t_fit = np.linspace(df_local["t_s"].min(), df_local["t_s"].max(), 2000)
    y_fit = A * np.sin(w0 * t_fit + phi) + B
    plt.plot(t_fit, y_fit, "k-", linewidth=2, label="Sine fit")
    for idx, p in enumerate(peaks_idx):
        t_val = df_local.loc[p, "t_s"]
        plt.axvline(x=t_val, color="red", linestyle="--", alpha=0.6, label="MZI peaks" if idx == 0 else None)
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

mzi_results, mzi_peaks_idx = estimate_mzi_fsr(df["t_s"], df["ch2_v"])

print("\n===== MZI FSR analysis =====")
print(f"FSR (time)       ≈ {mzi_results['fsr_time_s']:.6e} s")
print(f"FSR (wavelength) ≈ {mzi_results['fsr_nm']:.6e} nm")
print(f"FSR (frequency)  ≈ {mzi_results['fsr_hz']:.6e} Hz  ({mzi_results['fsr_hz']/1e9:.3f} GHz)")
print(f"ΔL               ≈ {mzi_results['delta_L_m']:.6e} m  ({mzi_results['delta_L_m']*1e6:.3f} um)")

mzi_df = pd.DataFrame({"t_s": df["t_s"].to_numpy(), "v": df["ch2_v"].to_numpy()})
plot_with_peaks(mzi_df, mzi_peaks_idx, mzi_results["fit_params"], label="CH2 (MZI)")


#%% -------- Map time to wavelength using the MZI FSR --------

fsr_time_s = float(mzi_results["fsr_time_s"])    # MZI FSR in time [s]
mzi_fsr_nm = float(mzi_results["fsr_nm"])        # MZI FSR in wavelength [nm]
nm_per_s = mzi_fsr_nm / fsr_time_s               # effective sweep speed from MZI

t0 = df["t_s"].iloc[0]
df["lambda_nm"] = LAMBDA_START + nm_per_s * (df["t_s"] - t0)
df["freq_hz"] = C0 / (df["lambda_nm"] * 1e-9)


# Warning for truncated sweeps
if CUT_SIGNAL:
    print("\nWARNING: The signal doesn't cover the expected wavelength range so it is not possible to plot resonances vs wavelength accurately."
          "\nNevertheless, ring FSR is reliable since it is computed using the MZI periodicty as a ruler.")
else:
    plt.figure(figsize=(10, 4))
    plt.plot(df["lambda_nm"], df["ch1_v"], label="Ring resonances", alpha=0.6)
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Voltage [V]")
    plt.grid(True)
    plt.legend()
    plt.title("Data mapped to wavelength axis")
    plt.tight_layout()
    plt.show()


#%% -------- Ring resonances detection and FSR extraction on CH1 --------
# Cell detects ring dips and computes FSR using the MZI-derived wavelength scale.

# Ring resonance detection and plotting 
PEAK_PROMINENCE = 0.2          # min resonance depth to keep
MIN_SEP_FRACTION = 0.7          # keep peaks at least this fraction of the (robust) FSR apart
PLOT_PEAK_COUNT = 7             # number of resonances to display
PLOT_PEAK_START_INDEX = 0       # starting resonance index for plotting


def detect_ring_resonances(times, voltage, prominence, min_sep_fraction=0.5):
    """Find clean resonance minima with de-duplication of close peaks."""

    dips, _ = find_peaks(-voltage, prominence=float(prominence))
    if len(dips) == 0:
        raise RuntimeError("No resonances found. Adjust PEAK_PROMINENCE or check the input data.")

    if len(dips) >= 2:
        minima_times = times[dips]
        diffs = np.diff(minima_times)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            c1 = float(np.percentile(diffs, 20))
            c2 = float(np.percentile(diffs, 80))
            for _ in range(50):
                thr = 0.5 * (c1 + c2)
                mask = diffs <= thr
                new_c1 = float(np.mean(diffs[mask])) if np.any(mask) else c1
                new_c2 = float(np.mean(diffs[~mask])) if np.any(~mask) else c2
                if abs(new_c1 - c1) < 1e-9 and abs(new_c2 - c2) < 1e-9:
                    break
                c1, c2 = new_c1, new_c2
            mu_short, mu_long = (c1, c2) if c1 < c2 else (c2, c1)
            dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
            fsr_guess = float(mu_long)
            min_sep = max(3.0 * dt, float(min_sep_fraction) * fsr_guess)

            keep = []
            group_anchor_idx = int(dips[0])
            best_idx = group_anchor_idx
            best_val = float(voltage[best_idx])

            for i in range(1, len(dips)):
                cur_idx = int(dips[i])
                if (times[cur_idx] - times[group_anchor_idx]) < min_sep:
                    if float(voltage[cur_idx]) < best_val:
                        best_idx = cur_idx
                        best_val = float(voltage[cur_idx])
                else:
                    keep.append(best_idx)
                    group_anchor_idx = cur_idx
                    best_idx = cur_idx
                    best_val = float(voltage[cur_idx])
            keep.append(best_idx)
            dips = np.array(sorted(set(keep)), dtype=int)
    return dips


def compute_ring_fsr(times, dips, nm_per_s_scale, lambda_start,
                     mzi_fsr_time_s, mzi_fsr_nm):
    """Convert resonance spacing to FSR in time, nm, and Hz."""
    minima_times = times[dips]
    if len(minima_times) >= 2:
        fsr_time_samples = np.diff(minima_times)
        fsr_time = float(np.median(fsr_time_samples))
    else:
        fsr_time_samples = np.array([])
        fsr_time = np.nan

    t_reference = float(times[0])
    minima_lambda = lambda_start + nm_per_s_scale * (minima_times - t_reference)
    lambda_center = float(np.median(minima_lambda)) if len(minima_lambda) else np.nan
    if (np.isfinite(fsr_time) and np.isfinite(mzi_fsr_time_s)
            and np.isfinite(mzi_fsr_nm) and mzi_fsr_time_s > 0):
        fsr_lambda = (fsr_time / float(mzi_fsr_time_s)) * float(mzi_fsr_nm)
    else:
        fsr_lambda = np.nan
    if np.isfinite(lambda_center) and np.isfinite(fsr_lambda):
        fsr_frequency = C0 * (fsr_lambda * 1e-9) / ((lambda_center * 1e-9) ** 2)
    else:
        fsr_frequency = np.nan

    return minima_times, fsr_time_samples, fsr_time, fsr_lambda, fsr_frequency


def plot_selected_resonances(times, voltage, dips, start_index=0, count=5, lambdas_nm=None,
                             fsr_time=None, fsr_lambda=None, fsr_frequency=None):
    """Plot a subset of ring resonances for a quick visual check (x-axis in wavelength)."""
    start = int(np.clip(start_index, 0, max(len(dips) - 1, 0)))
    stop = min(start + count, len(dips))
    selected_indices = dips[start:stop]
    if len(selected_indices) == 0:
        raise RuntimeError("Peak selection is empty. Check start/count.")

    if len(selected_indices) >= 2:
        fsr_samples_indices = int(np.round(np.median(np.diff(selected_indices))))
    else:
        fsr_samples_indices = max(len(times) // 100, 10)
    pad = max(fsr_samples_indices // 2, 20)
    i0 = max(selected_indices[0] - pad, 0)
    i1 = min(selected_indices[-1] + pad, len(times) - 1)

    if lambdas_nm is None:
        raise RuntimeError("lambdas_nm is required to plot resonances vs wavelength.")

    plt.figure(figsize=(10, 4))
    plt.plot(lambdas_nm[i0:i1 + 1], voltage[i0:i1 + 1], alpha=0.6, label="Data")
    plt.scatter(lambdas_nm[selected_indices], voltage[selected_indices], color="blue", zorder=2, label="Resonance minima")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Voltage [V]")
    plt.title("Selected ring resonances")
    # Legend in the upper-left corner
    plt.legend(loc="upper left")
    plt.grid(True)

    # Parameter textbox in upper-right corner
    if fsr_time is not None and fsr_lambda is not None and fsr_frequency is not None:
        param_text = (
            f"FSR_t = {fsr_time:.3e} s\n"
            f"FSR_λ = {fsr_lambda:.3e} nm\n"
            f"FSR_f = {fsr_frequency:.3e} Hz"
        )
        plt.gca().text(
            0.98, 0.98, param_text,
            transform=plt.gca().transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.9)
        )

    plt.tight_layout()
    plt.show()


voltage = df["ch1_v"].to_numpy()
times = df["t_s"].to_numpy()
dips = detect_ring_resonances(times, voltage, PEAK_PROMINENCE, MIN_SEP_FRACTION)
minima_times, fsr_time_samples, fsr_time, fsr_lambda, fsr_frequency = compute_ring_fsr(
    times,
    dips,
    nm_per_s,
    LAMBDA_START,
    mzi_results["fsr_time_s"],
    mzi_results["fsr_nm"],
)

if PLOT_PEAK_COUNT > len(minima_times):
    print(f"\nWarning: only {len(minima_times)} resonances found, so PLOT_PEAK_COUNT={len(minima_times)} is selected now.")

plot_selected_resonances(
    times,
    voltage,
    dips,
    start_index=PLOT_PEAK_START_INDEX,
    count=PLOT_PEAK_COUNT,
    lambdas_nm=df["lambda_nm"].to_numpy(),
    fsr_time=fsr_time,
    fsr_lambda=fsr_lambda,
    fsr_frequency=fsr_frequency
)

print("\n===== Ring Resonator FSR (from CH1) =====")
print(f"Number of resonances found: {len(minima_times)}")
print(f"FSR time differences (s): {fsr_time_samples}")
print(f"FSR (time) median ≈ {fsr_time:.3g} s")
print(f"FSR (wavelength) ≈ {fsr_lambda:.3g} nm")
print(f"FSR (frequency) ≈ {fsr_frequency:.3e} Hz  ({fsr_frequency/1e9:.3f} GHz)\n")


#%% -------- Fit helpers --------
# Cell defines Lorentzian models and fit wrappers for reuse across fit modes.

RESONANCE_AT_1550 = True                            # set True to fit the resonance nearest to 1550 nm, if False use FIT_PEAK_INDEX
#FIT_PEAK_INDEX = 2                                 # index of the resonance to fit if RESONANCE_AT_1550 = False
FIT_PEAK_INDEX = int((len(minima_times) - 1) / 2)   # default: middle resonance when selecting manually
WINDOW_FIT_FSR_FRACTION = 0.8                       # window size in multiples of FSR around the resonance to fit the Lorentzian
WINDOW_PLOT_FSR_FRACTION = 3                        # window size in multiples of FSR around the resonance to plot data + fit


def lorentzian(f, a, tau, FSR, f_res):
    """Lorentzian model for ring resonator transmission."""
    num = (1.0 - a**2) * (1.0 - tau**2)
    denom = (1.0 - a * tau)**2 + 4.0 * a * tau * np.sin(np.pi * (np.asarray(f) - f_res) / FSR)**2
    return num / denom

def lorentzian_dip(f, a, tau, FSR, f_res):
    """Lorentzian dip model: T = 1 - L(f)."""
    return 1.0 - lorentzian(f, a, tau, FSR, f_res)

def lorentzian_dip_scale(f, a, tau, FSR, f_res, scale):
    """Affine voltage model: scale * (1 - L(f))."""
    return scale * lorentzian_dip(f, a, tau, FSR, f_res)

def plot_lorentzian_fit(x_data, x_fit, y_data, y_fit, a, tau):
    """Overlay data and Lorentzian fit within the window and show fitted params."""
    data_order = np.argsort(x_data)
    fit_order = np.argsort(x_fit)
    x_data = np.asarray(x_data)[data_order]
    y_data = np.asarray(y_data)[data_order]
    x_fit = np.asarray(x_fit)[fit_order]
    y_fit = np.asarray(y_fit)[fit_order]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x_data, y_data, label="Data", alpha=0.8)
    ax.plot(x_fit, y_fit, label="Lorentzian fit", linewidth=2, alpha=0.6)
    ax.ticklabel_format(style="plain", axis="x", useOffset=False)
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Voltage [V]")
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(loc="lower left")

    param_text = f"a = {a:.3f}\n" f"tau = {tau:.3f}"
    ax.text(
        0.98, 0.02, param_text,
        transform=ax.transAxes, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.9)
    )
    fig.tight_layout()
    plt.show()


if len(dips) == 0:
    raise RuntimeError("No dips detected; run resonance detection first.")

lambdas_nm = df["lambda_nm"].to_numpy()
freqs_hz = df["freq_hz"].to_numpy()

# When CUT_SIGNAL is True, the wavelength axis is not reliable, so RESONANCE_AT_1550 is set to False
if CUT_SIGNAL:
    print("\nWavelength axis is not reliable, so it is not possible to automatically select the resonance near 1550 nm.\n"
        "RESONANCE_AT_1550 is set to False and it is necessary to choose FIT_PEAK_INDEX manually.")
    RESONANCE_AT_1550 = False

if RESONANCE_AT_1550:  # Use the resonance whose wavelength is closest to 1550 nm
    lambdas_at_dips = lambdas_nm[dips]
    if not np.all(np.isfinite(lambdas_at_dips)):
        raise RuntimeError(
            "Non-finite wavelength values at resonance positions; cannot "
            "automatically select resonance near the target wavelength."
        )
    idx = int(np.argmin(np.abs(lambdas_at_dips - 1550.0)))
else:
    idx = int(np.clip(FIT_PEAK_INDEX, 0, len(dips) - 1))

f_res = float(freqs_hz[dips[idx]])

if not np.isfinite(fsr_frequency) or fsr_frequency <= 0:
    raise RuntimeError("Initial FSR is not finite; cannot proceed with fitting.")
FSR_init = float(fsr_frequency)

mask = (freqs_hz >= f_res - WINDOW_FIT_FSR_FRACTION * FSR_init) & (freqs_hz <= f_res + WINDOW_FIT_FSR_FRACTION * FSR_init)
if not np.any(mask):
    raise RuntimeError("Empty window around selected resonance; check FSR or indices.")
f_win = freqs_hz[mask]
v_win = voltage[mask]


#%% -------- Lorentzian fit ----------------
# Fit alpha, tau, resonance frequency and scale with fixed FSR

def model(f, a, tau, f_res_fit, scale):
    return lorentzian_dip_scale(f, a, tau, FSR_init, f_res_fit, scale)

# Initial guesses
a0 = 0.95
tau0 = 0.9
f_res0 = float(f_res)   # start from the resonance frequency found from the dips
scale_0 = float(np.max(np.abs(v_win)))

# Fit bounds
f_res_lower = f_res0 - WINDOW_FIT_FSR_FRACTION * FSR_init
f_res_upper = f_res0 + WINDOW_FIT_FSR_FRACTION * FSR_init
scale_min = 0.0
scale_max = float(np.max(np.abs(v_win)))

popt, pcov = curve_fit(
    model,
    f_win,
    v_win,
    p0=(a0, tau0, f_res0, scale_0),
    bounds=(
        [1e-6, 1e-6, f_res_lower, scale_min],
        [1.0, 1.0, f_res_upper, scale_max],
    ),
    maxfev=20000
)

a_fit, tau_fit, f_res_fit, scale_fit = [float(x) for x in popt]

residuals = v_win - model(f_win, a_fit, tau_fit, f_res_fit, scale_fit)
chi_sq = float(np.sum(residuals**2))
dof = max(len(v_win) - len(popt), 1)
chi_sq_red = chi_sq / dof

perr = np.sqrt(np.diag(pcov))
a_err = float(perr[0])
tau_err = float(perr[1])
f_res_err = float(perr[2])
scale_err = float(perr[3])

b = a_fit * tau_fit
gamma = (1 - b) / np.sqrt(b) * FSR_init / np.pi if (1 >= a_fit >= 0 and 1 >= tau_fit >= 0) else np.nan
finesse = FSR_init / gamma if gamma > 0 else np.nan
Q_factor = f_res_fit / gamma if gamma > 0 else np.nan


# Compute fit-window arrays using WINDOW_FIT_FSR_FRACTION ("narrow" window)
mask = (freqs_hz >= f_res_fit - WINDOW_FIT_FSR_FRACTION * FSR_init) & (freqs_hz <= f_res_fit + WINDOW_FIT_FSR_FRACTION * FSR_init)
f_win = freqs_hz[mask]
v_win = voltage[mask]
f_dense = np.linspace(
    f_res_fit - WINDOW_FIT_FSR_FRACTION * FSR_init,
    f_res_fit + WINDOW_FIT_FSR_FRACTION * FSR_init, 1000
)
v_fit_dense = model(f_dense, a_fit, tau_fit, f_res_fit, scale_fit)
lambda_window_nm = (C0 / f_win) * 1e9
lambda_fit_dense_nm = (C0 / f_dense) * 1e9

print("\n===== Lorentzian Fit Results =====")
print("Fitted parameters:")
print(f"  f_res  = {f_res_fit:.3e} ± {f_res_err:.3e} Hz  ({f_res_fit/1e9:.3f} GHz)")
print(f"  a      = {a_fit:.3f} ± {a_err:.3f}")
print(f"  tau    = {tau_fit:.3f} ± {tau_err:.3f}")
print(f"  scale  = {scale_fit:.3f} ± {scale_err:.3f}")
print(f"\nChi-square         = {chi_sq:.3f}")
print(f"Reduced chi-square = {chi_sq_red:.3e}  (dof = {dof})")
print(f"\nFWHM linewidth     = {gamma:.3e} Hz  ({gamma/1e6:.3f} MHz)")
print(f"Finesse            = {finesse:.3f}")
print(f"Q-factor           = {Q_factor:.3e}\n")

# Plot fit in the narrow window (fit window)
plot_lorentzian_fit(lambda_window_nm, lambda_fit_dense_nm, v_win, v_fit_dense, a_fit, tau_fit)

# Wider window around the same resonance to compare fit with neighboring resonances
mask_plot = (
    (freqs_hz >= f_res_fit - WINDOW_PLOT_FSR_FRACTION * FSR_init) &
    (freqs_hz <= f_res_fit + WINDOW_PLOT_FSR_FRACTION * FSR_init)
)

f_plot = freqs_hz[mask_plot]
v_plot = voltage[mask_plot]

f_dense_plot = np.linspace(
    f_res_fit - WINDOW_PLOT_FSR_FRACTION * FSR_init,
    f_res_fit + WINDOW_PLOT_FSR_FRACTION * FSR_init,
    2000
)
v_fit_dense_plot = model(f_dense_plot, a_fit, tau_fit, f_res_fit, scale_fit)

lambda_plot_nm = (C0 / f_plot) * 1e9
lambda_fit_plot_nm = (C0 / f_dense_plot) * 1e9

plot_lorentzian_fit(lambda_plot_nm,lambda_fit_plot_nm,v_plot,v_fit_dense_plot,a_fit,tau_fit)



