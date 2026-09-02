# DepthWizard (SIH 2026)

RGB image → 3D terrain reconstruction & flythrough.

## Project Structure

This monorepo is divided into several main components:

- `/backend` — Core Python API (FastAPI) handling depth estimation and scale calibration.
- `/backend/analysis` — Slope computation, terrain classification, and DSM comparison modules.
- `/frontend` — Web interface (Three.js + JS) for uploading images and rendering the 3D flythrough.
- `/data` — Directory for sample images, DEM tiles, and cached demo assets.
- `/eval` — Accuracy evaluation harness and configuration.
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

## Analysis Modules

### Slope Computation

- **Method:** numpy gradient (central differences, edge_order=1)
- **Formula:** `slope_deg = arctan(sqrt(dz_dx² + dz_dy²))`
- **Output:** degrees [0, 90], percent grade [0, ∞)
- **NaN handling:** input NaN → output NaN; gradient computed on NaN-free substitute
- **Limitation:** 2-pixel boundary effects from finite differences; not suitable for sub-pixel analysis

### Terrain Classification

Slope-based categories (uint8 output):

| ID  | Category   | Slope Range |
| --- | ---------- | ----------- |
| 0   | flat       | < 2°        |
| 1   | gentle     | 2–8°        |
| 2   | moderate   | 8–15°       |
| 3   | steep      | 15–30°      |
| 4   | very_steep | > 30°       |
| 255 | nodata     | NaN input   |

### DSM Comparison (Reference Evaluation)

Compares predicted vs reference DSMs (must be spatially aligned):

- **Metrics:** RMSE, MAE, bias, Pearson correlation, median/P90/P95 absolute error
- **NaN masking:** pixels where either input is NaN are excluded
- **Degenerate cases:** returns None for metrics when < 2 valid pixels or zero variance

### API Endpoints

- `POST /slope` — Compute slope + classification from uploaded image
- `POST /classify` — Terrain classification from uploaded image
- `POST /evaluate` — Compare predicted DSM against reference (file paths in outputs/)

## Limitations

- Slope from relative depth reflects shape only, not true metric slope
- Demo heightmap is 8-bit (256 levels); 16-bit inputs preserve full precision
- No real-world evaluation data included; `eval/config.yaml` provides a template
- `transformers`/`torch` not installed in default environment — depth estimation requires manual install

## Documentation

- [Technical Summary](docs/TECHNICAL_SUMMARY.md)
- [Demo Script](docs/DEMO_SCRIPT.md)
