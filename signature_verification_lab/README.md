# Signature Verification — Lab

**Objective:** decide whether a handwritten signature is **genuine** or **forged** by comparing it with a few reference signatures of the same person. Signatures can be drawn on a tablet or uploaded as a photo/scan.

No dataset and no training are needed: each person's acceptance threshold is calibrated from their own references.

## Setup (macOS/Linux)
```bash
cd signature_verification_lab
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0
```
Open `http://HOST_IP:8501` on a tablet connected to the same Wi-Fi. Do not expose the port publicly.

## How to use
1. **Enroll** — choose or create a person and save at least 3 genuine signatures (5 is better).
2. **Verify** — pick the claimed identity, draw or upload a signature, press *Verify*.
3. **Profiles** — review the references and delete outliers.

## How it works
1. `preprocessing.py` — extracts the ink (Otsu threshold for photos/scans), crops it, scales it into a 128×256 frame and blurs it slightly.
2. `verification.py` — describes the signature with three equally weighted feature groups:
   HOG (stroke orientations), a 4×8 ink-density grid (layout) and projection profiles (stroke rhythm).
3. Calibration — leave-one-out distances between references give the natural variation of that person;
   the threshold is `max(worst case, mean + 2·std) × 1.15`.
4. Verification — mean distance from the query to the references. Score = `100 · 2^(-distance / threshold)`, so **≥ 50 means genuine**.

References are stored as normalized PNGs in `profiles/<name>/`.

## Limits
- This is a classical teaching baseline. It reliably rejects *different* signatures, but a **skilled forgery**
  (someone carefully copying the real shape) can pass. Do not use it for real security or legal decisions.
- It only checks the final image (offline verification). Pen speed and pressure are not used.
- Possible improvements: train a Siamese CNN on a public dataset (CEDAR, GPDS, BHSig260) and replace `extract_features` with the learned embedding.
