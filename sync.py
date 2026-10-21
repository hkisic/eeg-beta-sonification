"""
EEG <-> task alignment with cued blinks.

The myWaves Pebble has no trigger input, so EEG and task run on separate clocks. At the
start of every task session the word BLINK is shown three times, 2 s apart; the task logs
the task-clock time of each cue (``*_sync.csv``) and each blink leaves a large artefact in
Fp1/Fp2. The three blink positions are marked by hand in EDFbrowser as annotations whose
text contains "blink", and the file is saved. A straight line fitted through the three
(task time, EEG sample) pairs maps any task time to an EEG sample.
"""

import numpy as np
import pandas as pd

GAP_TOLERANCE_S = 0.25   # blink gaps on the two clocks must agree within this
SLOPE_TOLERANCE = 0.05   # fitted slope must be within 5 % of sfreq / 1000


def read_blink_annotations(raw, expected_n=3):
    """Sample numbers of the manually placed blink annotations, sorted in time."""
    ann = raw.annotations
    if len(ann) == 0:
        raise ValueError("No annotations in this EDF - mark the blinks in EDFbrowser and save the file.")
    onsets = np.asarray(ann.onset)
    is_blink = np.array(["blink" in d.lower() for d in ann.description])
    chosen = np.sort(onsets[is_blink] if is_blink.any() else onsets)
    if len(chosen) != expected_n:
        raise ValueError(f"Found {len(chosen)} blink annotations, expected {expected_n} "
                         f"(onsets at {np.round(chosen, 2)} s)")
    return (chosen * raw.info["sfreq"]).round().astype(int)


def read_sync_times(sync_csv):
    """Task-clock times (ms) of the blink cues, from the task's *_sync.csv."""
    return pd.read_csv(sync_csv).sort_values("blink_number")["task_clock_ms"].to_numpy(float)


def fit_alignment(blink_samples, task_ms, sfreq,
                  gap_tolerance_s=GAP_TOLERANCE_S, slope_tolerance=SLOPE_TOLERANCE):
    """
    Linear map task_clock_ms -> EEG sample, with sanity checks.
    Returns np.polyfit coefficients (slope in samples/ms, intercept in samples).
    """
    task_ms = np.asarray(task_ms, float)
    if len(task_ms) != len(blink_samples):
        raise ValueError(f"{len(task_ms)} sync cues but {len(blink_samples)} blinks")
    eeg_gaps = np.diff(blink_samples) / sfreq
    task_gaps = np.diff(task_ms) / 1000.0
    if np.any(np.abs(eeg_gaps - task_gaps) > gap_tolerance_s):
        raise ValueError(f"Blink gaps do not match: EEG {eeg_gaps} s vs task {task_gaps} s")
    coeffs = np.polyfit(task_ms, blink_samples, 1)
    expected = sfreq / 1000.0
    if abs(coeffs[0] - expected) / expected > slope_tolerance:
        raise ValueError(f"Fitted slope {coeffs[0]:.5f} differs >{slope_tolerance:.0%} "
                         f"from the expected {expected:.5f} samples/ms")
    return coeffs


def task_ms_to_sample(coeffs, t_ms):
    return int(np.polyval(coeffs, t_ms))

