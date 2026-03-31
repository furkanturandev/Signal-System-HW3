"""
COE216 Signals and Systems 2025-2026 Spring Semester
MidTerm Project - Sound Signal Analysis and Gender Classification
Using Time-Domain Autocorrelation Method

Single-file Streamlit application.
Run with:  streamlit run speech_gender_classifier.py

Requirements:
    pip install streamlit librosa numpy scipy pandas matplotlib seaborn openpyxl soundfile
"""

import os
import glob
import warnings
import tempfile
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
import streamlit as st

warnings.filterwarnings("ignore")


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

# Windowing: 20-30 ms windows during which signal can be considered stationary
FRAME_DURATION_MS = 25       # 25 ms frame (within required 20-30 ms)
HOP_DURATION_MS = 10         # 10 ms hop size

# F0 search range for autocorrelation
F0_MIN = 50                  # Hz - lowest expected F0
F0_MAX = 600                 # Hz - highest expected F0

# Voiced region detection thresholds
ENERGY_PERCENTILE = 30       # frames above this -> voiced candidate
ZCR_PERCENTILE = 70          # frames below this -> voiced candidate

# Rule-based classification thresholds (Hz)
#   Male:   ~85 - 165 Hz
#   Woman:  ~165 - 255 Hz
#   Child:  ~255 - 400+ Hz
MALE_UPPER = 165
WOMAN_UPPER = 255


# ============================================================================
# 2. DATA READING & METADATA
# ============================================================================
# Following the instructor's "Data Reading" Code Template:
#
#   import pandas as pd
#   import librosa
#   import glob
#
#   # Combine all excel files to create a master list
#   excel_files = glob.glob("Dataset/**/Grup_*.xlsx", recursive=True)
#   master_df = pd.concat([pd.read_excel(f) for f in excel_files])
#
#   # Loop for processing files
#   for index, row in master_df.iterrows():
#       audio, sr = librosa.load(row['File_Path'])
#       # Time domain analysis (e.g., Autocorrelation)
#       # f0 = compute_autocorrelation_pitch(audio, sr)
#
# Tips from instructor: Read metadata using pandas.read_excel() and then
# check for the existence of the audio file using os.path.exists().
# ============================================================================

def load_master_metadata(dataset_root):
    """
    Combine all group Excel files to create a master list.
    Normalizes column names to handle variations across groups.
    """
    excel_files = glob.glob(os.path.join(dataset_root, "**", "Grup_*.xlsx"), recursive=True)
    if not excel_files:
        excel_files = glob.glob(os.path.join(dataset_root, "**", "Group_*.xlsx"), recursive=True)
    if not excel_files:
        excel_files = glob.glob(os.path.join(dataset_root, "**", "*.xlsx"), recursive=True)

    if not excel_files:
        return pd.DataFrame()

    all_dfs = []
    for f in excel_files:
        try:
            df = pd.read_excel(f)

            # Strip whitespace from column names
            df.columns = [str(c).strip() for c in df.columns]

            # Normalize column names to a standard format
            col_map = {}
            for c in df.columns:
                cl = c.lower().replace(" ", "_")
                if cl in ["file_name", "file_path", "filename", "dosya_adi", "dosya_adı"]:
                    col_map[c] = "File_Name"
                elif cl in ["gender", "cinsiyet"]:
                    col_map[c] = "Gender"
                elif cl in ["age", "yas", "yaş"]:
                    col_map[c] = "Age"
                elif cl in ["subject_id", "denek_id"]:
                    col_map[c] = "Subject_ID"
                elif cl in ["feeling", "duygu"]:
                    col_map[c] = "Feeling"
                elif cl in ["sentence_no", "cumle_no", "cümle_no"]:
                    col_map[c] = "Sentence_No"

            df = df.rename(columns=col_map)
            df["_source_folder"] = os.path.dirname(f)
            all_dfs.append(df)
        except Exception:
            pass

    if not all_dfs:
        return pd.DataFrame()

    master_df = pd.concat(all_dfs, ignore_index=True)
    return master_df


def resolve_audio_path(row, dataset_root):
    """
    Find the actual .wav file. Column is normalized to 'File_Name'.
    """
    if "File_Name" not in row.index or pd.isna(row["File_Name"]):
        return None

    file_name = str(row["File_Name"]).strip()

    # 1) Direct path
    if os.path.exists(file_name):
        return file_name

    # 2) Relative to dataset root
    full_path = os.path.join(dataset_root, file_name)
    if os.path.exists(full_path):
        return full_path

    # 3) In the same folder as the Excel file
    if "_source_folder" in row.index:
        folder_path = os.path.join(row["_source_folder"], file_name)
        if os.path.exists(folder_path):
            return folder_path

        just_name = os.path.basename(file_name)
        folder_path2 = os.path.join(row["_source_folder"], just_name)
        if os.path.exists(folder_path2):
            return folder_path2

    # 4) Search all subfolders
    just_name = os.path.basename(file_name)
    for root, dirs, files in os.walk(dataset_root):
        if just_name in files:
            return os.path.join(root, just_name)

    return None


def get_gender_label(row):
    """
    Extract gender label from normalized 'Gender' column.
    Normalizes to: "Male", "Woman", "Child"
    """
    if "Gender" not in row.index or pd.isna(row["Gender"]):
        return "Unknown"

    value = str(row["Gender"]).strip().lower()

    if value in ["male", "erkek", "m", "e"]:
        return "Male"
    elif value in ["female", "woman", "kadın", "kadin", "f", "k", "w"]:
        return "Woman"
    elif value in ["child", "çocuk", "cocuk", "c", "ç"]:
        return "Child"
    else:
        return value.capitalize()


# ============================================================================
# 3. TIME-DOMAIN ANALYSIS
# ============================================================================
# Step 1: Preprocessing and Characterization in the Time Domain
#   - Windowing: 20-30 ms
#   - Short-Term Energy and ZCR for voiced region detection
#   - F0 analysis only on voiced regions
# ============================================================================

def compute_frames(sr, frame_ms=FRAME_DURATION_MS, hop_ms=HOP_DURATION_MS):
    """Compute frame and hop lengths in samples from ms values."""
    frame_length = int(sr * frame_ms / 1000)
    hop_length = int(sr * hop_ms / 1000)
    return frame_length, hop_length


def compute_short_term_energy(signal, frame_length, hop_length):
    """
    Short-Term Energy (STE) for each frame.
    STE = sum of x[n]^2
    Used for voiced region detection.
    """
    n_frames = 1 + (len(signal) - frame_length) // hop_length
    energy = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_length
        frame = signal[start:start + frame_length]
        energy[i] = np.sum(frame ** 2)
    return energy


def compute_zcr_per_second(signal, sr, frame_length, hop_length):
    """
    Zero Crossing Rate (ZCR): Number of zero-passes per second of the signal.
    (As specified: "Average ZCR: Number of zero-passes per second of the signal.")
    """
    n_frames = 1 + (len(signal) - frame_length) // hop_length
    zcr = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_length
        frame = signal[start:start + frame_length]
        signs = np.sign(frame)
        signs[signs == 0] = 1
        crossings = np.sum(np.abs(np.diff(signs)) > 0)
        # Convert to zero-crossings per second
        frame_duration_sec = frame_length / sr
        zcr[i] = crossings / frame_duration_sec
    return zcr


def detect_voiced_regions(energy, zcr):
    """
    Detect voiced regions using Energy and ZCR.
    Voiced speech: HIGH energy, LOW ZCR.
    Unvoiced/Silence: LOW energy or HIGH ZCR.
    """
    energy_threshold = np.percentile(energy, ENERGY_PERCENTILE)
    zcr_threshold = np.percentile(zcr, ZCR_PERCENTILE)
    return (energy > energy_threshold) & (zcr < zcr_threshold)


# ============================================================================
# 4. AUTOCORRELATION F0 ESTIMATION (Primary Method - REQUIRED)
# ============================================================================
# Step 2: Determining the Fundamental Frequency F0
# Autocorrelation Method (Time Domain):
#   R(tau) = sum of x[n] * x[n - tau]
# The lag tau at the highest peak -> T0 = tau -> F0 = sr / T0
# "In this assignment, you will be using the Autocorrelation Function."
# ============================================================================

def autocorrelation_f0_frame(frame, sr, f0_min=F0_MIN, f0_max=F0_MAX):
    """
    Compute F0 for a single frame using Autocorrelation.
    R(tau) = sum of x[n] * x[n - tau]
    F0 = sr / tau_peak
    """
    lag_min = int(sr / f0_max)   # shortest period (highest freq)
    lag_max = int(sr / f0_min)   # longest period (lowest freq)

    if lag_max >= len(frame):
        lag_max = len(frame) - 1
    if lag_min >= lag_max:
        return 0.0

    # Compute autocorrelation R(tau)
    frame_centered = frame - np.mean(frame)
    autocorr = np.correlate(frame_centered, frame_centered, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]  # positive lags only

    # Normalize by R(0)
    if autocorr[0] == 0:
        return 0.0
    autocorr = autocorr / autocorr[0]

    # Search for the highest peak in the valid lag range
    search_region = autocorr[lag_min:lag_max + 1]
    if len(search_region) < 3:
        return 0.0

    peaks, _ = find_peaks(search_region, height=0.2)
    if len(peaks) == 0:
        return 0.0

    # Select peak with highest autocorrelation value
    best_peak_idx = peaks[np.argmax(search_region[peaks])]
    best_lag = best_peak_idx + lag_min

    if best_lag == 0:
        return 0.0

    return sr / best_lag


def estimate_f0_autocorrelation(signal, sr, frame_length, hop_length, voiced_mask):
    """
    Estimate F0 for all voiced frames using Autocorrelation method.
    F0 analysis is ONLY performed on voiced regions (as required).
    """
    n_frames = len(voiced_mask)
    f0_per_frame = np.zeros(n_frames)
    f0_values = []

    for i in range(n_frames):
        if not voiced_mask[i]:
            continue
        start = i * hop_length
        end = start + frame_length
        if end > len(signal):
            break

        # Apply Hamming window
        frame = signal[start:end] * np.hamming(frame_length)
        f0 = autocorrelation_f0_frame(frame, sr)
        if f0 > 0:
            f0_per_frame[i] = f0
            f0_values.append(f0)

    return f0_values, f0_per_frame


# ============================================================================
# 5. FFT F0 ESTIMATION (For comparison / report only)
# ============================================================================
# "The magnitude spectrum |X(f)| is calculated.
#  The fundamental frequency is determined as the first dominant harmonic."
# Required for: Visual Comparison (Autocorrelation vs FFT side-by-side)
# ============================================================================

def estimate_f0_fft_frame(frame, sr, f0_min=F0_MIN, f0_max=F0_MAX):
    """
    FFT method: |X(f)| magnitude spectrum, first dominant harmonic = F0.
    """
    windowed = frame * np.hamming(len(frame))
    n_fft = max(4096, 2 ** int(np.ceil(np.log2(len(windowed))) + 1))
    spectrum = np.abs(np.fft.rfft(windowed, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    valid = (freqs >= f0_min) & (freqs <= f0_max)
    if not np.any(valid):
        return 0.0

    valid_spectrum = spectrum[valid]
    valid_freqs = freqs[valid]
    peaks, _ = find_peaks(valid_spectrum, height=np.max(valid_spectrum) * 0.3)

    if len(peaks) == 0:
        return valid_freqs[np.argmax(valid_spectrum)]

    # First dominant harmonic
    return valid_freqs[peaks[0]]


def estimate_f0_fft(signal, sr, frame_length, hop_length, voiced_mask):
    """Estimate F0 for all voiced frames using FFT."""
    n_frames = len(voiced_mask)
    f0_values = []
    f0_per_frame = np.zeros(n_frames)
    for i in range(n_frames):
        if not voiced_mask[i]:
            continue
        start = i * hop_length
        end = start + frame_length
        if end > len(signal):
            break
        frame = signal[start:end]
        f0 = estimate_f0_fft_frame(frame, sr)
        if f0 > 0:
            f0_per_frame[i] = f0
            f0_values.append(f0)
    return f0_values, f0_per_frame


# ============================================================================
# 6. FEATURE EXTRACTION (per audio file)
# ============================================================================
# B. Feature Extraction - Calculate for each audio file:
#   - Average F0 (Pitch): autocorrelation, voiced parts only, averaged
#   - Average ZCR: Number of zero-passes per second of the signal
#   - Energy Distribution: Amplitude characteristics of vocal regions
# ============================================================================

def extract_features(audio_path):
    """
    Extract all required features from a single audio file.
    Uses librosa.load(row['File_Path']) as in the instructor's template
    (without forcing sr, so original sample rate is preserved).
    """
    # Load audio - no sr override, preserving original sample rate
    signal, sr = librosa.load(audio_path, sr=None, mono=True)

    # Trim silence
    signal, _ = librosa.effects.trim(signal, top_db=25)
    if len(signal) == 0:
        return None

    # Windowing: 20-30 ms frames
    frame_length, hop_length = compute_frames(sr)

    # Short-Term Energy (STE)
    energy = compute_short_term_energy(signal, frame_length, hop_length)

    # Zero Crossing Rate (per second)
    zcr = compute_zcr_per_second(signal, sr, frame_length, hop_length)

    # Detect voiced regions using Energy + ZCR
    voiced_mask = detect_voiced_regions(energy, zcr)

    if np.sum(voiced_mask) < 5:
        # Relax threshold if too few voiced frames
        voiced_mask = energy > np.percentile(energy, 20)

    # F0 estimation using Autocorrelation (only on voiced regions)
    f0_values, f0_per_frame = estimate_f0_autocorrelation(
        signal, sr, frame_length, hop_length, voiced_mask
    )

    # Outlier removal using median filter
    if len(f0_values) > 0:
        med = np.median(f0_values)
        f0_filtered = [f for f in f0_values if med * 0.5 <= f <= med * 2.0]
        if len(f0_filtered) < 3:
            f0_filtered = f0_values
    else:
        f0_filtered = []

    # Average F0: mean of F0 values for voiced parts of one speaker
    avg_f0 = np.mean(f0_filtered) if f0_filtered else 0
    std_f0 = np.std(f0_filtered) if f0_filtered else 0

    # Average ZCR: zero-passes per second
    avg_zcr = np.mean(zcr)

    # Energy Distribution: amplitude characteristics of vocal regions
    avg_energy = np.mean(energy[voiced_mask]) if np.any(voiced_mask) else np.mean(energy)

    return {
        "avg_f0": avg_f0,
        "std_f0": std_f0,
        "avg_zcr": avg_zcr,
        "avg_energy": avg_energy,
        "f0_values": f0_filtered,
        "signal": signal,
        "sr": sr,
        "energy": energy,
        "zcr": zcr,
        "voiced_mask": voiced_mask,
        "frame_length": frame_length,
        "hop_length": hop_length,
        "f0_per_frame": f0_per_frame,
    }


# ============================================================================
# 7. RULE-BASED GENDER CLASSIFIER
# ============================================================================
# "develop your own algorithm based on the extracted features"
# "rule-based algorithm" for gender classification (Male, Woman, Child)
# ============================================================================

def classify_gender(avg_f0, avg_zcr=None, avg_energy=None,
                    male_upper=MALE_UPPER, woman_upper=WOMAN_UPPER):
    """
    Rule-based classification using F0:
        Male:   F0 < male_upper Hz
        Woman:  male_upper <= F0 < woman_upper Hz
        Child:  F0 >= woman_upper Hz
    """
    if avg_f0 <= 0:
        return "Unknown"
    if avg_f0 < male_upper:
        return "Male"
    elif avg_f0 < woman_upper:
        return "Woman"
    else:
        return "Child"


# ============================================================================
# 8. STREAMLIT UI
# ============================================================================
# "Interface: User interface design is flexible. You can create a
#  Web Interface (Streamlit, Flask)..."
# "Functionality: predict the class of an audio file when selected
#  and report overall accuracy across the entire dataset."
# ============================================================================

def main():
    st.set_page_config(page_title="COE216 - Speech Gender Classifier", page_icon="🎤", layout="wide")

    st.markdown("""
    <style>
    .main-title { text-align:center; font-size:2.2rem; font-weight:700; margin-bottom:0.2rem; }
    .sub-title  { text-align:center; font-size:1.1rem; color:#888; margin-bottom:2rem; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-title">🎤 Sound Signal Analysis & Gender Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">COE216 Signals & Systems 2025-2026 — Time-Domain Autocorrelation Method</div>', unsafe_allow_html=True)

    # ---- SIDEBAR ----
    st.sidebar.header("⚙️ Settings")

    st.sidebar.subheader("📂 Dataset Folder")
    dataset_path = st.sidebar.text_input(
        "Full path to your dataset folder:",
        value="",
        placeholder="C:/Users/.../Midterm_Dataset_2026",
    )
    st.sidebar.caption("💡 Paste the path to the folder containing Group_01, Group_02, ...")

    if dataset_path and os.path.isdir(dataset_path):
        subfolders = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
        wav_count = len(glob.glob(os.path.join(dataset_path, "**", "*.wav"), recursive=True))
        xlsx_count = len(glob.glob(os.path.join(dataset_path, "**", "*.xlsx"), recursive=True))
        st.sidebar.success("✅ Folder found!")
        st.sidebar.write(f"📁 {len(subfolders)} subfolder  |  🎵 {wav_count} .wav  |  📊 {xlsx_count} .xlsx")
        if subfolders:
            with st.sidebar.expander("Show subfolders"):
                for sf in subfolders[:30]:
                    st.write(f"📁 {sf}")
    elif dataset_path:
        st.sidebar.error("❌ Folder not found. Check the path.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎚️ Classification Thresholds")
    male_thresh = st.sidebar.slider("Male upper F0 (Hz)", 100, 250, MALE_UPPER)
    woman_thresh = st.sidebar.slider("Woman upper F0 (Hz)", 150, 400, WOMAN_UPPER)
    st.sidebar.caption(f"Male: F0 < {male_thresh} Hz\nWoman: {male_thresh}–{woman_thresh} Hz\nChild: F0 ≥ {woman_thresh} Hz")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**COE216** Signals & Systems")
    st.sidebar.markdown("Midterm Project 2025-2026")

    # ---- TABS ----
    tab1, tab2, tab3 = st.tabs([
        "📁 Single File Prediction",
        "📊 Full Dataset Analysis",
        "🔬 Autocorrelation vs FFT",
    ])

    # ================== TAB 1: SINGLE FILE ==================
    with tab1:
        st.header("Single File Prediction")
        st.caption("Upload any .wav file to predict its gender class (Male / Woman / Child).")

        uploaded = st.file_uploader("Upload a .wav file", type=["wav"], key="single")

        if uploaded is not None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            with st.spinner("Analyzing audio..."):
                feats = extract_features(tmp_path)

            if feats is not None:
                pred = classify_gender(feats["avg_f0"], feats["avg_zcr"], feats["avg_energy"],
                                       male_thresh, woman_thresh)

                pred_icons = {"Male": "🔵", "Woman": "🔴", "Child": "🟢", "Unknown": "⚪"}
                st.markdown(f"### {pred_icons.get(pred, '')} Prediction: **{pred}**")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Average F0 (Hz)", f"{feats['avg_f0']:.1f}")
                c2.metric("Std F0 (Hz)", f"{feats['std_f0']:.1f}")
                c3.metric("Avg ZCR (/sec)", f"{feats['avg_zcr']:.1f}")
                c4.metric("Voiced Frames", f"{int(np.sum(feats['voiced_mask']))}")

                # Plots
                signal = feats["signal"]
                sr = feats["sr"]
                hop = feats["hop_length"]
                fl = feats["frame_length"]
                t_sig = np.arange(len(signal)) / sr
                t_fr = np.arange(len(feats["energy"])) * hop / sr

                fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
                fig.suptitle(f"Time-Domain Analysis — {uploaded.name}", fontweight='bold')

                axes[0].plot(t_sig, signal, color='steelblue', linewidth=0.5)
                for i, v in enumerate(feats["voiced_mask"]):
                    if v:
                        axes[0].axvspan(i * hop / sr, i * hop / sr + fl / sr, alpha=0.12, color='green')
                axes[0].set_ylabel("Amplitude")
                axes[0].set_title("Waveform (green = voiced regions)")

                axes[1].plot(t_fr, feats["energy"], color='darkorange')
                axes[1].axhline(np.percentile(feats["energy"], ENERGY_PERCENTILE), color='red', ls='--', alpha=0.6, label='Threshold')
                axes[1].set_ylabel("Energy")
                axes[1].set_title("Short-Term Energy (STE)")
                axes[1].legend()

                axes[2].plot(t_fr, feats["zcr"], color='purple')
                axes[2].axhline(np.percentile(feats["zcr"], ZCR_PERCENTILE), color='red', ls='--', alpha=0.6, label='Threshold')
                axes[2].set_ylabel("ZCR (/sec)")
                axes[2].set_title("Zero Crossing Rate — zero-passes per second")
                axes[2].legend()

                f0d = feats["f0_per_frame"].copy()
                f0d[f0d == 0] = np.nan
                axes[3].plot(t_fr[:len(f0d)], f0d, 'o-', color='red', ms=2, lw=0.8)
                axes[3].set_ylabel("F0 (Hz)")
                axes[3].set_xlabel("Time (s)")
                axes[3].set_title(f"F0 Contour (Autocorrelation, voiced only) — Avg: {feats['avg_f0']:.1f} Hz")
                axes[3].set_ylim([F0_MIN, F0_MAX])

                for ax in axes:
                    ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.error("Could not extract features. File may be too short or silent.")

            os.unlink(tmp_path)

    # ================== TAB 2: DATASET ANALYSIS ==================
    with tab2:
        st.header("Full Dataset Analysis")

        if not dataset_path or not os.path.isdir(dataset_path):
            st.info("👈 Enter your dataset folder path in the sidebar first.")
        else:
            if st.button("🚀 Run Analysis on Entire Dataset", type="primary"):
                master_df = load_master_metadata(dataset_path)

                if master_df.empty:
                    st.error("No metadata Excel files found in the dataset folder.")
                else:
                    total = len(master_df)
                    st.write(f"Found **{total}** records in metadata. Processing...")

                    # Debug: show column names and first row
                    with st.expander("🔍 Debug: Excel structure"):
                        st.write("**Columns:**", master_df.columns.tolist())
                        st.write("**First row:**")
                        st.write(master_df.iloc[0].to_dict())

                    progress = st.progress(0, text="Starting...")
                    results = []
                    skipped = []

                    for idx, row in master_df.iterrows():
                        progress.progress((idx + 1) / total, text=f"Processing {idx+1}/{total}...")

                        audio_path = resolve_audio_path(row, dataset_path)
                        true_label = get_gender_label(row)

                        if audio_path is None or not os.path.exists(str(audio_path)):
                            if len(skipped) < 5:
                                fn = row.get("File_Name", "N/A")
                                skipped.append(f"{fn} (source: {row.get('_source_folder', 'N/A')})")
                            continue

                        try:
                            feats = extract_features(audio_path)
                            if feats is None:
                                continue

                            pred = classify_gender(feats["avg_f0"], feats["avg_zcr"],
                                                   feats["avg_energy"], male_thresh, woman_thresh)

                            results.append({
                                "File": os.path.basename(audio_path),
                                "True_Label": true_label,
                                "Predicted": pred,
                                "Avg_F0": round(feats["avg_f0"], 2),
                                "Std_F0": round(feats["std_f0"], 2),
                                "Avg_ZCR": round(feats["avg_zcr"], 2),
                                "Avg_Energy": round(feats["avg_energy"], 6),
                                "Correct": pred == true_label,
                            })
                        except Exception as e:
                            st.warning(f"Error processing {os.path.basename(str(audio_path))}: {e}")

                    progress.empty()

                    if not results:
                        st.error("No files processed.")
                        if skipped:
                            st.warning("**First skipped files (could not find):**")
                            for s in skipped:
                                st.write(f"- `{s}`")
                            st.info("Check that the file names in Excel match actual .wav files in the group folders.")
                    else:
                        rdf = pd.DataFrame(results)

                        # Overall Accuracy
                        acc = rdf["Correct"].mean() * 100
                        correct_n = int(rdf["Correct"].sum())

                        st.markdown("---")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Total Files Processed", len(rdf))
                        m2.metric("Correct Predictions", f"{correct_n}/{len(rdf)}")
                        m3.metric("Overall Accuracy", f"{acc:.1f}%")

                        # Statistics Table (matching instructor's format exactly)
                        # Class | Number of Samples | Average F0 (Hz) | Standard Deviation | Success (%)
                        st.subheader("📊 Statistical Table")
                        stats_rows = []
                        for cls in ["Male", "Woman", "Child"]:
                            cd = rdf[rdf["True_Label"] == cls]
                            n = len(cd)
                            stats_rows.append({
                                "Class": cls,
                                "Number of Samples": n,
                                "Average F0 (Hz)": f"{cd['Avg_F0'].mean():.2f}" if n > 0 else "—",
                                "Standard Deviation": f"{cd['Avg_F0'].std():.2f}" if n > 0 else "—",
                                "Success (%)": f"%{cd['Correct'].mean()*100:.1f}" if n > 0 else "—",
                            })
                        st.table(pd.DataFrame(stats_rows))

                        # Confusion Matrix
                        st.subheader("Confusion Matrix")
                        classes = ["Male", "Woman", "Child"]
                        valid = rdf[rdf["True_Label"].isin(classes)]
                        cm = np.zeros((3, 3), dtype=int)
                        c2i = {c: i for i, c in enumerate(classes)}
                        for _, r in valid.iterrows():
                            if r["True_Label"] in c2i and r["Predicted"] in c2i:
                                cm[c2i[r["True_Label"]]][c2i[r["Predicted"]]] += 1

                        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                    xticklabels=classes, yticklabels=classes, ax=ax_cm)
                        ax_cm.set_xlabel("Predicted Label")
                        ax_cm.set_ylabel("True Label")
                        ax_cm.set_title("Confusion Matrix")
                        plt.tight_layout()
                        st.pyplot(fig_cm)
                        plt.close()

                        # F0 Distributions
                        st.subheader("F0 Distributions by Class")
                        fig_d, axes_d = plt.subplots(1, 3, figsize=(14, 4))
                        colors_map = {"Male": "steelblue", "Woman": "salmon", "Child": "mediumseagreen"}
                        for i, cls in enumerate(classes):
                            data = rdf[rdf["True_Label"] == cls]["Avg_F0"]
                            if len(data) > 0:
                                axes_d[i].hist(data, bins=15, color=colors_map[cls], edgecolor='black', alpha=0.75)
                                axes_d[i].axvline(data.mean(), color='red', ls='--', label=f'Mean={data.mean():.0f} Hz')
                                axes_d[i].legend()
                            axes_d[i].set_title(f"{cls} (n={len(data)})")
                            axes_d[i].set_xlabel("F0 (Hz)")
                            axes_d[i].set_ylabel("Count")
                            axes_d[i].grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig_d)
                        plt.close()

                        # Error Analysis
                        errors = rdf[~rdf["Correct"]]
                        if len(errors) > 0:
                            st.subheader("❌ Error Analysis")
                            st.write(f"**{len(errors)}/{len(rdf)}** files misclassified")
                            st.dataframe(
                                errors[["File", "True_Label", "Predicted", "Avg_F0"]].reset_index(drop=True),
                                use_container_width=True,
                            )
                            st.markdown("""
**Possible causes of misclassification:**
- **Overlapping F0 ranges:** Woman and Child speakers can have very similar F0, especially for young women or older children.
- **Background noise:** Noise corrupts autocorrelation peaks, leading to incorrect F0 estimation.
- **Emotional state:** Excitement or stress raises F0; sadness or fatigue lowers it, shifting a speaker into an adjacent class.
- **Speaking style:** Whispering produces very low energy making voiced detection unreliable; shouting can raise F0 unnaturally.
- **Short recording duration:** Fewer voiced frames means less reliable F0 averaging.
- **Tone of voice:** Some male speakers naturally speak at higher pitch, and some female speakers at lower pitch.
                            """)
                        else:
                            st.success("🎉 Perfect accuracy — no misclassifications!")

                        with st.expander("Show all individual results"):
                            st.dataframe(rdf, use_container_width=True)

                        csv = rdf.to_csv(index=False)
                        st.download_button("⬇️ Download Results CSV", csv, "classification_results.csv", "text/csv")

    # ================== TAB 3: AC vs FFT ==================
    # "Visual Comparison: Plot the Autocorrelation graph and the FFT Spectrum
    #  side-by-side for a selected subject's audio recording."
    with tab3:
        st.header("Autocorrelation vs FFT Comparison")
        st.caption("Upload a .wav to compare Autocorrelation and FFT F0 estimation side-by-side (for report).")

        comp_file = st.file_uploader("Upload .wav file for comparison", type=["wav"], key="compare")

        if comp_file is not None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(comp_file.read())
                tmp_path = tmp.name

            with st.spinner("Computing Autocorrelation & FFT..."):
                feats = extract_features(tmp_path)

            if feats is not None:
                signal = feats["signal"]
                sr = feats["sr"]
                fl = feats["frame_length"]
                hl = feats["hop_length"]
                vm = feats["voiced_mask"]

                f0_fft_vals, _ = estimate_f0_fft(signal, sr, fl, hl, vm)
                avg_f0_fft = np.mean(f0_fft_vals) if f0_fft_vals else 0

                c1, c2, c3 = st.columns(3)
                c1.metric("Autocorrelation Avg F0", f"{feats['avg_f0']:.1f} Hz")
                c2.metric("FFT Avg F0", f"{avg_f0_fft:.1f} Hz")
                c3.metric("Difference", f"{abs(feats['avg_f0'] - avg_f0_fft):.1f} Hz")

                vi = np.where(vm)[0]
                if len(vi) > 0:
                    mid = vi[len(vi) // 2]
                    start = mid * hl
                    end = start + fl
                    frame = signal[start:end]
                    windowed = frame * np.hamming(fl)

                    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
                    fig.suptitle(f"Autocorrelation vs FFT — {comp_file.name}", fontweight='bold', fontsize=14)

                    # Time domain
                    t_ms = np.arange(len(frame)) / sr * 1000
                    axes[0, 0].plot(t_ms, frame, color='steelblue', lw=0.8)
                    axes[0, 0].set_title("Time Domain (Selected Voiced Frame)")
                    axes[0, 0].set_xlabel("Time (ms)")
                    axes[0, 0].set_ylabel("Amplitude")
                    axes[0, 0].grid(True, alpha=0.3)

                    # Autocorrelation R(tau)
                    fc = windowed - np.mean(windowed)
                    ac = np.correlate(fc, fc, mode='full')
                    ac = ac[len(ac) // 2:]
                    if ac[0] != 0:
                        ac = ac / ac[0]
                    lags_ms = np.arange(len(ac)) / sr * 1000
                    lag_max_d = min(int(sr / F0_MIN) + 50, len(ac))
                    axes[0, 1].plot(lags_ms[:lag_max_d], ac[:lag_max_d], color='darkorange')
                    f0_ac = autocorrelation_f0_frame(windowed, sr)
                    if f0_ac > 0:
                        axes[0, 1].axvline(1000 / f0_ac, color='red', ls='--', label=f'F0 = {f0_ac:.0f} Hz')
                        axes[0, 1].legend()
                    axes[0, 1].set_title("Autocorrelation R(τ)")
                    axes[0, 1].set_xlabel("Lag τ (ms)")
                    axes[0, 1].set_ylabel("Normalized R(τ)")
                    axes[0, 1].grid(True, alpha=0.3)

                    # FFT |X(f)|
                    n_fft = 4096
                    spec = np.abs(np.fft.rfft(windowed, n=n_fft))
                    freqs = np.fft.rfftfreq(n_fft, d=1 / sr)
                    spec_db = 20 * np.log10(spec + 1e-10)
                    mask = freqs <= 1000
                    axes[1, 0].plot(freqs[mask], spec_db[mask], color='green', lw=0.8)
                    f0_f = estimate_f0_fft_frame(windowed, sr)
                    if f0_f > 0:
                        axes[1, 0].axvline(f0_f, color='red', ls='--', label=f'F0 = {f0_f:.0f} Hz')
                        axes[1, 0].legend()
                    axes[1, 0].set_title("FFT Magnitude Spectrum |X(f)|")
                    axes[1, 0].set_xlabel("Frequency (Hz)")
                    axes[1, 0].set_ylabel("Magnitude (dB)")
                    axes[1, 0].grid(True, alpha=0.3)

                    # Summary box
                    axes[1, 1].axis('off')
                    summary = (
                        f"F0 Comparison Results\n{'=' * 32}\n\n"
                        f"Autocorrelation F0:  {f0_ac:.1f} Hz\n"
                        f"FFT F0:              {f0_f:.1f} Hz\n\n"
                        f"Difference:          {abs(f0_ac - f0_f):.1f} Hz\n"
                        f"Relative Error:      {abs(f0_ac - f0_f) / max(f0_ac, 0.1) * 100:.1f}%\n\n"
                        f"Frame Duration:      {FRAME_DURATION_MS} ms\n"
                        f"Sampling Rate:       {sr} Hz\n"
                        f"Frame Samples:       {fl}"
                    )
                    axes[1, 1].text(0.1, 0.5, summary, transform=axes[1, 1].transAxes,
                                    fontsize=12, va='center', fontfamily='monospace',
                                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                    st.info(
                        "The F0 values from Autocorrelation and FFT should be approximately equal. "
                        "Small differences arise from spectral resolution and peak detection algorithms. "
                        "The autocorrelation peak at lag τ corresponds to the first dominant harmonic in the FFT spectrum."
                    )
                else:
                    st.warning("No voiced frames detected in this recording.")
            else:
                st.error("Could not process this file.")

            os.unlink(tmp_path)


if __name__ == "__main__":
    main()
