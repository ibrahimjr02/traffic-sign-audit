# Traffic Sign Visibility Audit

This project is a small, reproducible audit of real traffic-sign photographs from the **German Traffic Sign Recognition Benchmark (GTSRB)**.
I measure simple image properties with OpenCV, build a transparent rule-based **review-priority flag** with NumPy, and summarise the results with pandas and Matplotlib.

- **Course:** CSC 605 Advanced AI Programming, University of Tabuk
- **Student:** Salihu Umar Ibrahim (ID 481009445)
- **Central question:** *Which images should be reviewed first, and what evidence supports that priority?*

> **Reading the results.** Every cutoff in this project is a *project-specific triage setting*, not a road-safety standard. A review flag is a triage aid, not a safety verdict. Sharpness (variance of the Laplacian) is a proxy, not a human readability score. Associations are not causal claims.

## Dataset and citation

- Dataset: German Traffic Sign Recognition Benchmark (GTSRB), Real-Time Computer Vision Group, Ruhr-Universität Bochum.
- Official information: <https://benchmark.ini.rub.de/gtsrb_dataset.html>
- Archival copy used: <https://zenodo.org/records/13741936>

The dataset is **not included** in this repository and is not redistributed. I use only the labeled **training** images and their supplied annotations.

### Data layout

The dataset comes from the Zenodo record above (`data.zip`) and is extracted into `data/`, which gives this layout:

```
traffic-sign-audit/
├── data/                    <- the downloaded dataset (not tracked by git)
│   ├── Train.csv            <- one row per training image (used)
│   ├── signnames.csv        <- class ID -> sign name (used)
│   ├── Train/0 ... Train/42 <- training images, one folder per class (used)
│   └── Test/, Test.csv, Meta/, Meta.csv   (not used)
├── notebooks/traffic_sign_audit.ipynb
├── src/traffic_audit.py
├── outputs/figures/         <- saved figures (created by a run)
├── outputs/processed/       <- saved tables (created by a run)
├── requirements.txt
└── README.md
```

Notes on this copy of the data:
- The images are PNG files. `Train.csv` is comma-separated with the columns `Width, Height, Roi.X1, Roi.Y1, Roi.X2, Roi.Y2, ClassId, Path`. The loader also accepts semicolon-separated files and other capitalisation of the ROI columns.
- The `Path` column starts with `/code/...`, which is the dataset author's folder. The code removes that prefix and builds every path relative to `data/`, so no personal absolute path is used or stored.

## Setup

Python 3.10 or newer is required.

**Windows (PowerShell)**
```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1          # if script execution is blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt
pip install ipykernel               # only needed for the notebook
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install ipykernel               # only needed for the notebook
```

## Running the project

- **Notebook.** `notebooks/traffic_sign_audit.ipynb` runs from the first cell to the last (Run All in VS Code or Jupyter, with the `.venv` environment as the kernel).
- **Script.** `python src/traffic_audit.py`, started from the repository root with the environment active, recreates all tables in `outputs/processed/` and all figures in `outputs/figures/` from the raw annotations. The option `--seed` changes the random seed.

## What I did

1. **Load and validate** `Train.csv`. I check data types, missing values, duplicates, invalid classes, invalid ROI boxes and missing image files. Nothing is dropped silently: every excluded row and its reason is saved to `outputs/processed/exclusion_log.csv`.
2. **Sample reproducibly.** I choose eight classes by a fixed rule (evenly spaced ranks of class size) and draw at most 500 images per class with seed 42. Full-dataset counts always use all annotation rows and are kept separate from the sampled analysis.
3. **Measure** each sampled image with OpenCV (`cv2.imread`, BGR to grayscale):

| Feature | Definition |
|---|---|
| brightness | mean grayscale intensity (0-255) |
| contrast | standard deviation of grayscale intensity |
| sharpness | variance of `cv2.Laplacian` (a proxy, not a readability score) |
| img_width, img_height, aspect_ratio, min_side | size of the loaded image in pixels |
| roi_brightness, roi_contrast, roi_sharpness | the same measures on the annotated region of interest |

4. **Flag images for review** with vectorised NumPy percentile cutoffs. An image earns a point for each condition: *dark*, *overexposed*, *low contrast*, *blurry*, *small*. Score 1 is medium priority and score 2 or more is high priority. The base rule uses the 10th and 90th percentiles; a sensitivity check repeats it with the 5th/95th and 20th/80th.
5. **Summarise and plot** with pandas `groupby` and Matplotlib only (no Seaborn).

## Code structure

| Item | Purpose |
|---|---|
| `load_annotations` | reads and normalises the annotation CSV |
| `validate_annotations` | flags invalid rows without deleting them |
| `extract_image_features` | reads one image and measures it; failures are returned, not raised |
| `assign_review_priority` | applies the vectorised percentile rule with `np.select` |
| `TrafficSignAudit` | class that organises load, validate, sample, measure, flag, summarise, plot and save |

The notebook ends with a validation cell. It checks that an image-read failure is caught, that the feature columns exist, that the review flags contain only `True`/`False`, that every analysed image traces to an annotation row and a real file, and that no absolute path is stored.

## Outputs

- `outputs/processed/`: validation checks, exclusion log, skipped images, full-dataset class counts, the tidy per-image feature table, the review priority list, the sensitivity summary, flag rates by class, class profiles, associations, the ROI comparison, a track summary (flags within physical signs) and example images (CSV).
- `outputs/figures/`: class counts, brightness and sharpness distributions, class-wise boxplots, flag rate by class, size and contrast by class, sharpness associations, a flagged/unflagged contact sheet, three rule examples and a false-alarm/missed-case check (PNG).

## Main findings

- All 39,209 training annotation rows passed validation, and all 3,260 sampled images from eight classes could be read.
- Under the base rule (10th/90th percentiles) 32.0% of the sampled images are flagged and 10.5% are high priority. The strict and loose settings flag 15.2% and 57.6%.
- Classes 42, 4 and 3 have the highest flag rates under all three settings.
- Flags for darkness, low contrast and blur overlap strongly, and the sharpness measure partly repeats darkness and contrast.
- Images flagged only as overexposed were readable in all six random cases I inspected, so I rank them last.
- Flags cluster within physical signs: the 3,260 images come from 256 sign tracks.

## Limitations

- My analysis uses a sample of at most 500 images per selected class, not the full training set.
- GTSRB training images come in sequences of the same physical sign, so images from one sequence are not independent.
- Sharpness depends on image resolution, which varies a lot in GTSRB.
- The images do not represent every local road environment.
- Percentile cutoffs describe this sample and would shift with a different sample.
