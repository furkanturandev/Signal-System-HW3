# Speech Gender Classifier

This project extracts the fundamental frequency (F0) from speech signals using a time-domain autocorrelation method and predicts the speaker's gender class (Male / Woman / Child) with a simple rule-based classifier.

Files
- `speech_gender_classifier.py` — Single-file Streamlit application containing the analysis code and user interface.

Requirements
- Python 3.8+ (3.9 or 3.10 recommended)
- The required Python packages are listed in `requirements.txt`.

Installation (Windows PowerShell)

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run

Start the Streamlit interface with:

```powershell
streamlit run speech_gender_classifier.py
```

Usage summary
- Enter the full path to your dataset folder in the sidebar. That folder should contain group subfolders such as `Group_01`, `Group_02`, etc.
- The code looks for Excel metadata files in this order:
  1. `**/Grup_*.xlsx`
 2. `**/Group_*.xlsx`
 3. `**/*.xlsx` (if the above patterns are not found)
- Column names in Excel files are normalized. Important columns include:
  - `File_Name` (or equivalents) — file name or path to the .wav file
  - `Gender` — values may be "Male", "Woman", "Child" or their Turkish equivalents
- The app supports single-file predictions (upload a .wav) and full dataset analysis using Excel metadata.

Notes about Excel and .wav files
- Values in the Excel `File_Name` column can be either full paths or just file names. The code checks these locations in order:
  - The exact path if specified in the cell
  - Joined with the dataset root path
  - The folder where the Excel file is located
  - A search through all dataset subfolders for a matching file name
- If the .wav files cannot be found, the corresponding records are skipped and the first few skipped examples are shown to the user.

Troubleshooting
- If you see "No metadata Excel files found", confirm the dataset folder path and that Excel files use `Grup_` or `Group_` naming (or include at least one `.xlsx`).
- Very short or silent .wav files may not be analyzable.
- If you encounter errors related to `openpyxl` or `librosa`, verify that required packages are installed.

Developer notes
- Primary F0 estimation is performed using autocorrelation; FFT-based estimation is also computed for comparison and visual reporting.
- Classification thresholds (MALE_UPPER, WOMAN_UPPER) are defined in `speech_gender_classifier.py` and can be adjusted via the Streamlit sidebar.

License
- No specific license is included. Add a license appropriate for your use.

Contact
- If you want further changes or additions, I can update the repository accordingly.