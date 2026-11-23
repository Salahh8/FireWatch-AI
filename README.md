# FireWatch AI

Satellite-Based Wildfire Hotspot Analysis Using Big Data and Apache Spark.

## Project Overview

FireWatch AI analyzes satellite thermal detections across Australia during 2024 using Apache Spark and Machine Learning.

The project explores temporal patterns, geographic hotspots, fire radiative power (FRP), and monthly thermal detection prediction.

## Live Dashboard

https://firewatch-ai-jarrkdvrjrbrrjcl9e2uv5.streamlit.app/

## Dataset

| Item | Description |
|---|---|
| Source | NASA FIRMS |
| Satellite | VIIRS S-NPP |
| Collection | 2 |
| Region | Australia |
| Period | January - December 2024 |
| Original records | 979,867 |
| Cleaned records | 968,740 |
| Processed columns | 19 |

Data source: https://firms.modaps.eosdis.nasa.gov/download/

## Technologies

- Python 3.11
- Apache Spark 4.0.1
- PySpark
- Spark SQL
- Spark MLlib
- Pandas
- NumPy
- PyArrow
- Matplotlib
- Plotly
- Streamlit
- Jupyter Notebook

## Project Workflow

1. Data collection
2. Data quality assessment
3. Data cleaning and preprocessing
4. Spark filtering and aggregation
5. Exploratory data analysis
6. Geographic hotspot analysis
7. K-Means clustering and evaluation
8. Spark geographic join
9. Prediction dataset preparation
10. Random Forest training and evaluation
11. Geographic prediction
12. Interactive dashboard

## Data Preprocessing

- Check missing values and duplicate records.
- Validate geographic coordinates.
- Filter unwanted thermal detection types.
- Extract year, month, and day of year.
- Create FRP categories.
- Save cleaned data in Parquet format.

### FRP Categories

| Category | FRP |
|---|---|
| Low | Below 10 MW |
| Moderate | 10 to below 50 MW |
| High | 50 MW or above |

These categories are defined for this project.

## Apache Spark

Spark is used for:

- Loading and processing large datasets.
- DataFrame filtering and transformations.
- GroupBy and aggregation.
- Spark SQL queries.
- Geographic LEFT JOIN.
- Parquet storage.
- Machine Learning using Spark MLlib.

## Exploratory Data Analysis

The analysis includes:

- Monthly detection counts.
- Monthly average FRP.
- Day and night observations.
- Satellite confidence categories.
- FRP distributions.
- Geographic hotspot maps.

### Key Findings

| Metric | Result |
|---|---:|
| Total cleaned detections | 968,740 |
| Peak detection month | October |
| October detections | 198,374 |
| Average FRP | 10.76 MW |
| Maximum FRP | 1,232.79 MW |
| Daytime detections | 644,987 |
| Nighttime detections | 323,753 |
| High FRP detections | 30,347 |

## Geographic Analysis

The project uses 1-degree latitude and longitude grid cells to analyze the spatial distribution of thermal detections.

Spark LEFT JOIN results:

- Input records: 968,740
- Joined records: 968,740
- Observed geographic cells: 702
- Unmatched region IDs: 0

The prediction stage uses a separate geographic grid containing 822 cells, including cells without recorded detections.

## Machine Learning

### 1. K-Means Clustering

K-Means groups satellite detections based on latitude and longitude.

Different values of K were evaluated using the silhouette score.

| K | Sample Silhouette Score |
|---|---:|
| 3 | 0.5685 |
| 4 | 0.6615 |
| 5 | 0.6155 |
| 6 | 0.6001 |

K = 4 was selected based on the sampled evaluation.

Full-data silhouette score: 0.6607.

### 2. Random Forest Prediction

The Random Forest model predicts whether a geographic grid cell will contain at least one satellite thermal detection during a selected month.

Input features:

- Latitude grid
- Longitude grid
- Month
- Previous month detection count
- Previous two months detection count
- Previous three months detection count

Target:

- 0: No recorded detection
- 1: At least one recorded detection

### Temporal Data Split

| Dataset | Period | Records |
|---|---|---:|
| Training | April - September 2024 | 4,932 |
| Validation | October 2024 | 822 |
| Testing | November - December 2024 | 1,644 |
| Total | April - December 2024 | 7,398 |

A chronological split is used instead of a random split.

### Model Evaluation

| Metric | Test Result |
|---|---:|
| Accuracy | 81.39% |
| Weighted Precision | 81.39% |
| Weighted Recall | 81.39% |
| Weighted F1 Score | 81.36% |
| ROC-AUC | 87.67% |

### Baseline Comparison

The persistence baseline predicts a detection when the previous month contains at least one detection.

| Model | Test Accuracy |
|---|---:|
| Persistence Baseline | 78.65% |
| Random Forest | 81.39% |

Random Forest improves accuracy by approximately 2.74 percentage points.

### January 2025 Scenario

The trained model was used to generate an experimental January 2025 scenario using October, November, and December 2024 observations.

- Geographic cells: 822
- Predicted detection cells: 447
- Predicted no-detection cells: 375

Actual January 2025 outcomes were not evaluated. This is an experimental retrospective scenario, not a live or validated future fire forecast.

## Interactive Dashboard

The dashboard contains six pages:

1. Overview
2. Temporal Analytics
3. Geographic Intelligence
4. ML Intelligence
5. Hotspot Prediction
6. Data Explorer

Dashboard features:

- Interactive charts and filters.
- Geographic hotspot maps.
- Satellite and 2D compatibility maps.
- K-Means cluster visualization.
- Random Forest model-score maps.
- Geographic cell inspection.
- Historical detection charts.
- CSV export.
- Cleaned dataset preview.

## Project Structure

    FireWatch-AI/
    |-- data/
    |   |-- raw/
    |   |-- processed/
    |       |-- analytics/
    |-- src/
    |-- notebooks/
    |   |-- notebook.ipynb
    |-- models/
    |-- dashboard/
    |   |-- app.py
    |-- requirements.txt
    |-- requirements-dev.txt
    |-- README.md

## Installation

Clone the repository:

    git clone https://github.com/Salahh8/FireWatch-AI.git
    cd FireWatch-AI

Create a Python 3.11 virtual environment:

    python -m venv .venv311

Activate the environment on Windows:

    .\.venv311\Scripts\Activate.ps1

Install dashboard dependencies:

    python -m pip install -r requirements.txt

For Spark analysis and notebook execution:

    python -m pip install -r requirements-dev.txt

Java 17 is required for Spark. Windows installations may also require Hadoop utilities and environment configuration.

## Run the Dashboard

    python -m streamlit run dashboard/app.py

The dashboard uses precomputed analytics and prediction files. It does not require Spark or the full raw dataset.

## Run the Notebook

Open:

    notebooks/notebook.ipynb

Select the Python environment and run the notebook cells in order.

The full analysis requires the original dataset and locally generated processed files.

## Reproduce the Analysis

Download the NASA FIRMS VIIRS S-NPP Collection 2 archive for Australia during 2024.

Place the original CSV in:

    data/raw/fire_archive_SV-C2_842285.csv

Use the numbered scripts in the src directory to reproduce preprocessing, analytics, clustering, geographic joins, prediction dataset preparation, model training, and prediction exports.

The raw dataset, full processed datasets, and trained model files are excluded from GitHub to keep the repository lightweight.

## Limitations

- The dataset covers one year only.
- Satellite thermal detections are not confirmed individual wildfire incidents.
- FRP does not directly measure total wildfire damage.
- K-Means uses geographic coordinates rather than geodesic distances.
- Model scores are not calibrated real-world wildfire probabilities.
- January 2025 predictions have not been externally validated.
- The platform does not provide live wildfire monitoring.

## Academic Information

- University: Egyptian Chinese University (ECU)
- Faculty: Computers and Information Systems
- Course: Big Data (INF3101)
- Supervisor: Dr. Mohammed Al-Gunaid

## Data Source

NASA Fire Information for Resource Management System (FIRMS).

https://firms.modaps.eosdis.nasa.gov/
