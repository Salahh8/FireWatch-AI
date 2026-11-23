
from pathlib import Path
import io
import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import box

BASE = Path(__file__).resolve().parents[1]

OUTPUT = BASE / "data" / "processed" / "australia_grid.parquet"
BOUNDARY_FILE = BASE / "data" / "raw" / "australia_boundary.geojson"

# Natural Earth Admin-0 country boundaries
URL = (
    "https://raw.githubusercontent.com/nvkelso/"
    "natural-earth-vector/master/geojson/"
    "ne_10m_admin_0_countries.geojson"
)

BOUNDARY_FILE.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

print("\n=== LOADING AUSTRALIA BOUNDARY ===")

if not BOUNDARY_FILE.exists():
    response = requests.get(URL, timeout=90)
    response.raise_for_status()
    BOUNDARY_FILE.write_bytes(response.content)

world = gpd.read_file(BOUNDARY_FILE)

# Natural Earth uses ADMIN for country names.
australia = world[world["ADMIN"] == "Australia"]

if australia.empty:
    raise ValueError("Australia boundary not found.")

geometry = australia.geometry.union_all()

print("Boundary loaded successfully.")

print("\n=== BUILDING 1-DEGREE GRID ===")

minx, miny, maxx, maxy = geometry.bounds

rows = []

for lat in range(int(miny) - 1, int(maxy) + 1):
    for lon in range(int(minx) - 1, int(maxx) + 1):

        cell = box(lon, lat, lon + 1, lat + 1)

        # Include cells intersecting Australian land.
        if geometry.intersects(cell):
            rows.append({
                "lat_grid": lat,
                "lon_grid": lon
            })

grid = pd.DataFrame(rows).drop_duplicates()

grid = grid.sort_values(
    ["lat_grid", "lon_grid"]
).reset_index(drop=True)

grid.to_parquet(OUTPUT, index=False)

print("Total Australian grid cells:", len(grid))
print("Saved:", OUTPUT)
print("\n=== SAMPLE ===")
print(grid.head(10).to_string(index=False))
print("\n=== COMPLETE ===")
