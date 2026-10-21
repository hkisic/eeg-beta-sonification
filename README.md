# Beta-wave sonification for sustained attention

Can listening to your own brain activity help you stay focused?

This project records frontal EEG while people do a vigilance task. It extracts the
short bursts of **beta activity** (15–25 Hz) associated with focused attention, turns
them into a **personalised sound sequence**, and plays that sound back to the same
person during a later session. The idea extends the personalised-sonification approach
that improved sleep (Aloulou et al., 2023) to waking attention.

M1 internship project by Helena Kisić, Integrative and Computational Neuroscience team,
Paris-Saclay Institute of Neuroscience (NeuroPSI), supervised by Prof. Alain Destexhe.

## How it works

```
 vigilance task + EEG  ->  beta motifs  ->  personalised sound  ->  task again, listening to it
   (tasks/)               (motifs.py)       (sonify.py)
                  \_ blink sync aligns EEG and task (sync.py), band power tracks brain state (eeg.py)
```

1. **Measuring attention.** Two pygame tasks in `tasks/`:
   - a Psychomotor Vigilance Task (PVT), where you react as fast as possible to a
     stimulus that appears at random times;
   - a Multiple-Object Monitoring task (Karimi-Rouzbahani et al., 2021), where you watch
     moving dots and detect the rare ones that don't turn away.

   Both log every trial with millisecond timestamps.

2. **Synchronising EEG and behaviour** (`sync.py`). The portable EEG device (myWaves
   Pebble, Fp1/Fp2) has no trigger input, so each task starts by asking for three blinks
   2 s apart. The blinks leave large artefacts in the EEG. Matching their positions to
   the logged cue times gives a linear map from task time to EEG sample.

3. **Measuring brain state** (`eeg.py`). The EEG goes through these steps:
   - band-pass filter at 1–40 Hz;
   - split into 2 s epochs and reject artefacts above 150 µV;
   - compute a Welch power spectrum;
   - take theta (5–8 Hz), alpha (9–13 Hz) and beta (15–25 Hz) power, plus the
     beta/alpha ratio and the engagement index β/(α+θ).

   This can be done for a whole session or for the window just before each stimulus.

4. **Finding beta motifs** (`motifs.py`). Beta activity comes in brief bursts rather
   than a steady rhythm. The beta-filtered signal is scanned with two thresholds
   (Destexhe & Foubert, 2022):
   - a motif starts when the signal rises above mean + 0.8 SD;
   - it ends when its running average drops below mean + 0.6 SD.

   Each motif is described by its onset, peak, offset, rise time and decay time.

5. **Turning motifs into sound** (`sonify.py`). Each motif becomes a 1 s piano-like
   tone placed at its original time:
   - **pitch:** taken from the motif's duration (f = 0.5/T), transposed into the audible range;
   - **loudness:** taken from its peak amplitude.

   The result keeps the timing and dynamics of the person's own beta activity. A
   scrambled control, with the same tones shuffled in time, is generated alongside.

## Usage

```bash
pip install -r requirements.txt

# run a task (writes trial, sync and summary files)
python tasks/pvt_task.py --participant P1 --condition no_music --out recordings/
python tasks/mom_task.py --participant P1 --audio music --out recordings/

# make a personalised sonification from an attentive recording
python -m sonification.motifs recordings/P1.edf --skip-seconds 60 --out P1_motifs.csv
python -m sonification.sonify P1_motifs.csv --out-dir sounds --name P1
```

The analysis functions can also be used directly:

```python
from sonification import eeg, sync

raw = eeg.load_edf("recordings/P1.edf")          # sync blinks marked as "blink" annotations
coeffs = sync.fit_alignment(sync.read_blink_annotations(raw),
                            sync.read_sync_times("recordings/pvt_P1_..._sync.csv"),
                            raw.info["sfreq"])
sample = sync.task_ms_to_sample(coeffs, stimulus_time_ms)   # where a stimulus falls in the EEG
_, _, _, powers = eeg.process_recording("recordings/P1.edf")
print(powers["beta_alpha_ratio"], powers["engagement"])
```

Notes on the recordings:
- The myWaves Pebble's EDF files need their header re-saved once in EDFbrowser before
  MNE can open them.
- Mark the three sync blinks in EDFbrowser as annotations containing the word `blink`.

## Status

This was a small pilot (4 participants for the PVT phase, 9 for the MOM phase), limited
by device failures. It established the full pipeline, but did not show a reliable effect
of sonification on attention yet. Next steps:
- larger samples;
- interleaved control conditions;
- a mean-field model of how a neural population responds to sound derived from its own
  activity.

No recordings are included: EEG data are personal health data.

## References

- Destexhe, A. & Foubert, L. (2022). A method to convert neural signals into sound sequences. *J. Acoust. Soc. Am.* 151(6):3685–3689.
- Aloulou, H. et al. (2023). Listening to the sound of your own brain waves enhances sleep quality and quantity. *J. Sleep Res.* 32(5):e13923.
- Karimi-Rouzbahani, H. et al. (2021). Neural signatures of vigilance decrements predict behavioural errors before they occur. *NeuroImage* 224:117405.
- Dinges, D. F. & Powell, J. W. (1985). Microcomputer analyses of performance on a portable, simple visual RT task during sustained operations. *Behav. Res. Methods* 17(6):652–655.
