# ReconX — AI-Assisted Reconnaissance & Cyber Risk Analyzer

A beginner-friendly educational project for basic DNS resolution, HTTPS confirmation, selected HTTP security-header checks, ML risk classification, and CSV reporting.

## Important scope and limitations
- Only assess systems you own or have explicit permission to test.
- This starter does **not** exploit vulnerabilities and does **not** perform port scanning.
- The Decision Tree is a real scikit-learn model, but it is trained on a small **synthetic educational dataset**. Its labels and displayed risk level are not validated security ratings.
- The numeric score is a simple heuristic; the ML label and score may differ because they use different logic.
- DNS resolution may return multiple addresses; this does not indicate that the addresses are vulnerable.

## Install (Windows)
1. Install Python 3.11 or newer from https://www.python.org/downloads/ and enable **Add Python to PATH**.
2. Extract this ZIP to a folder, then open that folder in VS Code.
3. Open Terminal in VS Code and run:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run this instead in the same terminal:
```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

The app opens in your browser. Enter an authorized website/hostname and click **Run basic assessment**.

## Project files
- `app.py` — Streamlit dashboard and report download
- `recon.py` — DNS lookup and basic HTTP configuration checks
- `ml_model.py` — Decision Tree training, testing, and prediction
- `training_data.csv` — synthetic training examples
- `requirements.txt` — Python dependencies

## Next improvements
Use a larger, documented dataset; add proper validation and explainability; build unit tests; and add authorized lab-only modules after the initial version is understood.
