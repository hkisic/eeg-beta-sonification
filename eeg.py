"""
EEG preprocessing and spectral band power.

Pipeline:
  EDF -> 1-40 Hz IIR band-pass -> non-overlapping 2 s epochs -> reject epochs with
  peak-to-peak > 150 uV -> Welch PSD (1-40 Hz) -> mean power per band.

Bands: theta 5-8 Hz, alpha 9-13 Hz, beta 15-25 Hz.

Note: EDF files from the myWaves Pebble only open in MNE after their header has been
re-saved once in EDFbrowser (open the file, fix the header, save a copy).
"""

import numpy as np
import mne

BANDS = {"theta": (5.0, 8.0), "alpha": (9.0, 13.0), "beta": (15.0, 25.0)}
L_FREQ, H_FREQ = 1.0, 40.0
EPOCH_S = 2.0
REJECT_V = 150e-6

def load_edf(path):
    """Read an EDF recording; all channels (Fp1, Fp2) are treated as EEG."""
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})
    return raw


def bandpass(raw, l_freq=L_FREQ, h_freq=H_FREQ):
    return raw.copy().filter(l_freq=l_freq, h_freq=h_freq, method="iir", verbose=False)


def clean_epochs(raw_filtered, duration=EPOCH_S, reject=REJECT_V):
    """Fixed-length epochs with amplitude rejection. Returns (epochs, n_epochs_before)."""
    epochs = mne.make_fixed_length_epochs(raw_filtered, duration=duration, preload=True, verbose=False)
    n_before = len(epochs)
    epochs.drop_bad(reject=dict(eeg=reject), verbose=False)
    return epochs, n_before


def welch_psd(epochs, fmin=L_FREQ, fmax=H_FREQ):
    """PSD of every epoch and channel: (psds [epochs, channels, freqs], freqs)."""
    return epochs.compute_psd(method="welch", fmin=fmin, fmax=fmax, verbose=False).get_data(return_freqs=True)


def band_mask(freqs, band):
    lo, hi = BANDS[band] if isinstance(band, str) else band
    return (freqs >= lo) & (freqs <= hi)


def band_power_per_epoch(psds, freqs, band):
    """Mean PSD in a band for each epoch (averaged over both channels)."""
    return psds[:, :, band_mask(freqs, band)].mean(axis=(1, 2))


def session_band_powers(psds, freqs):
    """
    Session-level band powers and indices from the epoch PSDs.

    Absolute powers are mean PSD over the band's bins, channels and epochs (V^2/Hz).
    Relative powers divide by the mean PSD over 1-40 Hz (a ratio of mean densities, so
    values above 100 % are possible for bands denser than the broadband average).
    """
    total = psds.mean()
    out = {"n_epochs": psds.shape[0]}
    for name in BANDS:
        p = psds[:, :, band_mask(freqs, name)].mean()
        out[name] = p
        out[f"{name}_rel"] = p / total if total > 0 else np.nan
    out["beta_alpha_ratio"] = out["beta"] / out["alpha"] if out["alpha"] > 0 else np.nan
    out["theta_beta_ratio"] = out["theta"] / out["beta"] if out["beta"] > 0 else np.nan
    denom = out["alpha"] + out["theta"]          # engagement index, Pope et al. (1995)
    out["engagement"] = out["beta"] / denom if denom > 0 else np.nan
    return out


def process_recording(path):
    """Full pipeline for one EDF file -> (epochs, psds, freqs, band-power dict)."""
    raw = load_edf(path)
    epochs, n_before = clean_epochs(bandpass(raw))
    if len(epochs) == 0:
        raise ValueError(f"No clean epochs left in {path}")
    psds, freqs = welch_psd(epochs)
    powers = session_band_powers(psds, freqs)
    powers["n_epochs_before"] = n_before
    return epochs, psds, freqs, powers



def window_band_powers(segment, sfreq):
    """
    Band powers of a short data segment of shape (channels, samples), e.g. the window
    just before a task stimulus, to relate momentary brain state to the response.
    """
    info = mne.create_info(segment.shape[0], sfreq, "eeg")
    psds, freqs = mne.io.RawArray(segment, info, verbose=False).compute_psd(
        method="welch", fmin=L_FREQ, fmax=H_FREQ, n_fft=min(segment.shape[1], int(sfreq)), verbose=False
    ).get_data(return_freqs=True)
    out = {name: psds[:, band_mask(freqs, name)].mean() for name in BANDS}
    out["beta_alpha_ratio"] = out["beta"] / out["alpha"] if out["alpha"] > 0 else np.nan
    denom = out["alpha"] + out["theta"]
    out["engagement"] = out["beta"] / denom if denom > 0 else np.nan
    return out
