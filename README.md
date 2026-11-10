
# FireWatch AI

### Satellite-Based Wildfire Detection and Hotspot Analysis Using Big Data and Apache Spark

FireWatch AI is a Big Data analytics project that explores satellite-detected thermal hotspots across Australia during 2024.

It combines Apache Spark, Spark MLlib, geographic analytics, and an interactive Streamlit dashboard to identify temporal and spatial patterns in satellite fire detections.

## Project Scope

- Region: Australia
- Period: January 1 - December 31, 2024
- Source: NASA FIRMS
- Satellite: VIIRS S-NPP
- Collection: 2
- Original records: 979,867
- Cleaned records: 968,740

Important: NASA FIRMS detections are satellite thermal anomalies, not independently verified wildfire events. FireWatch AI performs historical analysis, not real-time monitoring or future fire prediction.

## Technologies

- Python 3.11
- Apache Spark / PySpark 4.0.1
- Spark SQL
- Spark MLlib
- Pandas and PyArrow
- Plotly
- Streamlit

## Data Pipeline

1. Load NASA FIRMS satellite observations.
2. Inspect data quality and record types.
3. Clean and preprocess satellite detections.
4. Perform Spark filtering and aggregations.
5. Run exploratory data analysis using Spark SQL.
6. Generate geographic hotspot statistics.
7. Evaluate K-Means clustering configurations.
8. Train geographic K-Means clustering.
9. Perform geographic reference enrichment using Spark LEFT JOIN.
10. Export aggregated analytics for the dashboard.

## Machine Learning

Algorithm: Spark MLlib K-Means

- Selected number of clusters: 4
- Selection based on a sampled silhouette comparison
- Full-data silhouette score: approximately 0.6607
- Full-data records analyzed: 968,740

Clustering uses geographic coordinates to group spatial detections. These clusters are analytical groupings, not official geographic regions or confirmed wildfire boundaries.

## Key Findings

- October had the highest detection count: 198,374.
- February had the highest monthly average FRP: approximately 16.90 MW.
- Daytime detections: 644,987.
- Nighttime detections: 323,753.
- High-FRP category: 30,347 detections.

FRP categories are project-defined analytical thresholds and should not be interpreted as official wildfire severity classifications.

## Geographic Spark Join

The project performs a Spark LEFT JOIN between satellite detections and a geographic grid reference derived from the observations.

- Original records: 968,740
- Joined records: 968,740
- Geographic grid cells: 702
- Unmatched region IDs: 0

The reference uses 1-degree latitude/longitude grid cells, not Australian state boundaries.

## Interactive Dashboard

The Streamlit dashboard contains five pages:

1. Overview
2. Temporal Analytics
3. Geographic Intelligence
4. ML Intelligence
5. Data Explorer

It visualizes satellite detection counts, FRP distributions, temporal trends, geographic hotspots, and clustering results.

## Installation

Requirements:

- Python 3.11
- Java 17 for Apache Spark

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/FireWatch-AI.git
cd FireWatch-AI
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the Dashboard

```bash
python -m streamlit run dashboard/app.py
```

Open the local URL displayed by Streamlit, usually:

http://localhost:8501

The repository includes the small precomputed analytics files needed by the dashboard.

## Reproduce the Analysis

Download the relevant NASA FIRMS archive data from:

https://firms.modaps.eosdis.nasa.gov/download/

Place the expected CSV file in `data/raw/`.

Then execute the scripts in the `src/` directory in their numbered workflow order, starting with data loading and preprocessing.

The raw dataset and full cleaned dataset are excluded from GitHub to keep the repository lightweight.

## Limitations

- The dataset covers Australia during 2024 only.
- Satellite detections do not necessarily correspond to unique wildfire events.
- FRP is an observed radiative-power measurement, not a direct estimate of total wildfire damage.
- K-Means uses geographic coordinates rather than a geodesic distance metric.
- The selected cluster count was chosen using sampled evaluation, not exhaustive full-data optimization.
- The dashboard presents historical analytics, not live satellite updates.

## Data Source

NASA Fire Information for Resource Management System (FIRMS)

https://firms.modaps.eosdis.nasa.gov/

## Project Status

- Data collection: Completed
- Preprocessing: Completed
- Spark analytics and SQL: Completed
- Spark geographic join: Completed
- K-Means clustering: Completed
- Interactive dashboard: Completed
- Data validation: Passed
