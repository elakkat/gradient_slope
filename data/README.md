# EEG data setup (CHB-MIT)

`scripts/gradient_effect_mit_eeg.py` is the only script that needs data you
must obtain yourself. MNIST, Fashion-MNIST (via `torchvision`), and the
DistilBERT/SST-2/QQP data (via HuggingFace `datasets`) all auto-download on
first run into `data/mnist/`, `data/fmnist/`, and the HuggingFace cache
respectively — no action needed for those.

## Source

CHB-MIT Scalp EEG Database (Shoeb & Guttag), hosted on PhysioNet:
https://physionet.org/content/chbmit/1.0.0/

It is openly downloadable (Open Data Commons Attribution License, ODC-BY) —
no credentialed-access application is required, but you must credit the
original authors when using it (see the PhysioNet page for the citation).

## Why it's not bundled in this repo

The raw database is large (tens of GB across all subjects/channels), and this
paper only uses a specific preprocessed subset/format built from it, not the
raw EDF files directly. Rather than ship a multi-GB derived artifact (and to
keep attribution clean), this repo documents the exact preprocessing so you
can regenerate the same input from your own PhysioNet download.

## What the script expects

`gradient_effect_mit_eeg.py` loads a single **pickle file** with this schema
(see `load_and_preprocess()` in the script — this is the ground truth):

```python
{
  'samples': [
      {'data': <numpy array, shape (23, 5120)>, 'label': 0 or 1},
      ...
  ],
  # 'meta': {...}  (optional, unused by the script)
}
```

- `data`: 23-channel scalp EEG segment, 256 Hz sampling rate, 20-second
  window → 23 × (20 × 256) = 23 × 5120 samples.
- `label`: `1` = seizure, `0` = non-seizure (clinician-labelled).

An older two-key schema, `{'Data': [...], 'Label': [...]}` (parallel lists
instead of a list of dicts), is also supported.

To build this from the raw PhysioNet EDF files, you need to:
1. Extract 20-second windows from each subject's recordings, using the
   clinician-provided seizure annotation files (`*-summary.txt`) to label
   windows as seizure/non-seizure.
2. Select/average to a consistent 23-channel montage per window (the script
   itself averages across the 23 channels into a single trace — see step 2
   below — so channel ordering does not need to match exactly, only the
   count).
3. Save the resulting `{'samples': [...]}` structure as a pickle.

## What the script does with it (already implemented, no action needed)

1. Averages across the 23 channels → one 5120-sample trace per segment.
2. Downsamples 256 Hz → 32 Hz via FIR anti-aliasing decimation
   (`scipy.signal.decimate(..., ftype='fir', zero_phase=True)`, factor 8)
   → 640 samples per segment.
3. Z-scores each segment independently (zero mean, unit variance).
4. Stratified 75/25 train/test split, with minority-class oversampling
   applied to the training split only.

## Pointing the script at your file

Either drop your pickle at the default location:

```
data/chb-mit/MIT_SIENE_23_0_to_240_n802_50_25_25_clinician_labelled.pkl
```

(relative to the repo root — the exact filename doesn't matter if you use
the env var below), or set an environment variable to any path:

```bash
export CHB_MIT_PKL_PATH=/path/to/your/preprocessed_chbmit.pkl   # Linux/macOS
setx CHB_MIT_PKL_PATH "C:\path\to\your\preprocessed_chbmit.pkl"  # Windows
```

If the file isn't found, the script prints a clear error pointing back to
this file instead of failing with a raw traceback.
