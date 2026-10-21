"""
Beta-wave motif detection and parametrisation.

Two-threshold scanning procedure of Destexhe & Foubert (2022):
  * the recording goes through the standard pipeline (1-40 Hz, 2 s epochs, 150 uV
    rejection), the clean epochs are band-passed to beta (15-25 Hz), averaged over the two
    channels and concatenated into one trace, from which the mean is subtracted;
  * a motif starts when the trace crosses mean + 0.8 SD upwards and ends when a running
    average of the samples since onset falls below mean + 0.6 SD;
  * each motif is described by its onset, peak and offset (time and amplitude), its rise
    time (onset -> peak) and decay time (peak -> offset).

Usage
-----
    python -m sonification.motifs recording.edf --skip-seconds 83 --out P1_motifs.csv
"""

import argparse
import os

import numpy as np
import pandas as pd

from . import eeg

UPPER_SD, LOWER_SD = 0.8, 0.6
BETA = (15.0, 25.0)


def beta_trace(edf_path, skip_seconds=0.0):
    """
    Continuous beta-band trace used for motif detection.

    Returns (trace, dt_ms). ``skip_seconds`` drops the start of the trace (the part before
    the task, e.g. the sync blinks). Because rejected epochs are removed before the epochs
    are concatenated, times in the trace are "clean-signal time", not recording time.
    """
    raw = eeg.load_edf(edf_path)
    epochs, _ = eeg.clean_epochs(eeg.bandpass(raw))
    beta = epochs.copy().filter(l_freq=BETA[0], h_freq=BETA[1], method="iir", verbose=False)
    trace = beta.get_data().mean(axis=1).flatten()
    dt = 1000 / raw.info["sfreq"]
    trace = trace - np.mean(trace)
    return trace[int(skip_seconds * 1000 / dt):], dt


def detect_motifs(trace, dt, upper_sd=UPPER_SD, lower_sd=LOWER_SD):
    """Two-threshold motif detection. Returns (DataFrame with one row per motif, thresholds)."""
    trace = np.asarray(trace)
    upper = np.mean(trace) + upper_sd * np.std(trace)
    lower = np.mean(trace) + lower_sd * np.std(trace)
    rows = []
    n = len(trace)
    i, prev = 1, trace[0]
    while i < n:
        val = trace[i]
        if val > upper and prev < upper:
            onset_t, onset_x = dt * i, val
            locavg = peak_x = val
            peak_t = dt * i
            nlocal = 1
            while locavg > lower and i < n - 1:          # running average since onset
                i += 1
                locavg = (locavg * nlocal + trace[i]) / (nlocal + 1)
                nlocal += 1
                if trace[i] > peak_x:
                    peak_x, peak_t = trace[i], dt * i
            rows.append({"onset_ms": onset_t, "onset_amp": onset_x,
                         "peak_ms": peak_t, "peak_amp": peak_x,
                         "offset_ms": dt * i, "offset_amp": trace[i],
                         "rise_ms": peak_t - onset_t, "decay_ms": dt * i - peak_t})
        prev = val
        i += 1
    cols = ["onset_ms", "onset_amp", "peak_ms", "peak_amp", "offset_ms", "offset_amp", "rise_ms", "decay_ms"]
    return pd.DataFrame(rows, columns=cols), (upper, lower)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("edf")
    p.add_argument("--skip-seconds", type=float, default=0.0,
                   help="seconds to drop at the start (before the sync blinks); recording-specific")
    p.add_argument("--out", required=True, help="output CSV with one row per motif")
    a = p.parse_args()

    trace, dt = beta_trace(a.edf, a.skip_seconds)
    motifs, thresholds = detect_motifs(trace, dt)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    motifs.to_csv(a.out, index=False)
    print(f"{len(motifs)} beta motifs -> {a.out}")
    print(f"Mean rise time {motifs['rise_ms'].mean():.1f} ms, mean decay time {motifs['decay_ms'].mean():.1f} ms")


if __name__ == "__main__":
    main()
