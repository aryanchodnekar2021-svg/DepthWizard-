# DepthWizard (SIH 2026)

RGB image → 3D terrain reconstruction & flythrough.

## Project Structure

This monorepo is divided into several main components:
- `/backend` — Core Python API (FastAPI) handling depth estimation and scale calibration.
- `/frontend` — Web interface (Three.js + JS) for uploading images and rendering the 3D flythrough.
- `/data` — Directory for sample images, DEM tiles, and cached demo assets.
- `/eval` — Accuracy evaluation harness.
- `/docs` — Technical summaries and demo scripts.

## Quickstart

### 1. Setup Backend
```bash
cd backend
python -m venv venv

# Windows
.\venv\Scripts\activate
# Linux/Mac
# source venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --reload
```
The API will be available at `http://localhost:8000`.

### 2. Setup Frontend
In a new terminal window:
```bash
cd frontend
python -m http.server 8080
```
Open `http://localhost:8080` in your browser.

## Demo Mode
If you don't want to run the backend Python API, you can still explore the 3D flythrough by running just the frontend web server and selecting "Dummy Forest Region" from the demo dropdown menu.

## Documentation
- [Technical Summary](docs/TECHNICAL_SUMMARY.md)
- [Demo Script](docs/DEMO_SCRIPT.md)
