import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="FireWatch AI | Command Center", page_icon="🔥", layout="wide", initial_sidebar_state="expanded")
ROOT = Path(__file__).resolve().parent.parent
ANALYTICS = ROOT / "data" / "processed" / "analytics"
MODEL = ROOT / "models" / "kmeans_metadata.json"
BG, PANEL, PANEL2, EDGE = "#090D14", "#141B26", "#1A2330", "#293341"
TEXT, MUTED, ORANGE, RED, GOLD = "#F2F5F9", "#9DAABD", "#FF8A3D", "#FF514F", "#FFCC74"

st.markdown("""
<style>
:root {color-scheme: dark;}
html, body, [data-testid="stAppViewContainer"], .stApp {background:#090D14!important;color:#F2F5F9!important;}
[data-testid="stHeader"] {background:#090D14!important;}
[data-testid="stSidebar"], [data-testid="stSidebarContent"] {background:#111822!important;}
[data-testid="stSidebar"] * {color:#E9EDF3!important;}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {color:#A9B6C8!important;}
[data-testid="stAppViewBlockContainer"] {padding-top:2.1rem;max-width:1500px;}
[data-testid="stMetric"] {background:#141B26;border:1px solid #293341;border-radius:14px;padding:18px 20px;min-height:110px;}
[data-testid="stMetricLabel"] {color:#A9B6C8!important;}
[data-testid="stMetricValue"] {color:#F5F7FB!important;}
h1,h2,h3,h4,h5,h6,p,span,label {color:inherit;}
h1 {letter-spacing:-.045em!important;}
[data-testid="stMarkdownContainer"] p {line-height:1.6;}
[data-testid="stVerticalBlockBorderWrapper"] {border-color:#293341!important;}
[data-testid="stDataFrame"] {border:1px solid #293341;border-radius:12px;overflow:hidden;}
[data-testid="stSelectbox"], [data-testid="stSlider"], [data-testid="stCheckbox"] {color:#F2F5F9;}
[data-testid="stSidebar"] [role="radiogroup"] label {padding:9px 8px;border-radius:8px;}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {background:#202B39;}
.stButton button, .stDownloadButton button {background:#1A2330!important;color:#F2F5F9!important;border:1px solid #354154!important;border-radius:10px!important;}
.stButton button:hover, .stDownloadButton button:hover {border-color:#FF8A3D!important;color:#FF8A3D!important;}
hr {border-color:#293341!important;}
.fire-eyebrow {font-size:12px;font-weight:800;letter-spacing:.17em;color:#FF8A3D;margin-bottom:8px;}
.fire-title {font-size:clamp(28px,3.2vw,43px);font-weight:850;letter-spacing:-.045em;color:#F2F5F9;line-height:1.14;}
.fire-sub {color:#A4B1C2;font-size:14px;margin-top:8px;margin-bottom:24px;}
.fire-panel {background:#141B26;border:1px solid #293341;border-radius:14px;padding:17px 20px;margin:8px 0 16px;}
.fire-panel strong {color:#F6F7FA;}
.fire-panel p {color:#A9B6C8;margin:7px 0 0;}
.fire-brand {font-size:24px;font-weight:850;letter-spacing:-.04em;color:#F6F7FA;}
.fire-brand span {color:#FF8A3D;}
.fire-tag {display:inline-block;background:#34231C;color:#FFB27C;border:1px solid #68432C;border-radius:99px;padding:5px 12px;font-size:11px;font-weight:750;letter-spacing:.08em;}
/* Improve form labels readability */
[data-testid="stWidgetLabel"] *,
[data-testid="stRadio"] label *,
[data-testid="stSlider"] label *,
[data-testid="stSelectbox"] label *,
[data-testid="stToggle"] label * {
    color: #E9EDF3 !important;
}

[data-testid="stCaptionContainer"] {
    color: #A9B6C8 !important;
}
/* High contrast widget labels on the dark application shell. */
[data-testid="stWidgetLabel"] p,
[data-testid="stRadio"] p,
[data-testid="stCheckbox"] p,
[data-testid="stToggle"] p,
[data-testid="stSlider"] p,
[data-testid="stSelectbox"] p {
    color: #E9EDF3 !important;
    opacity: 1 !important;
}
/* The selectbox's dropdown is a light surface in some themes: keep its text readable. */
[data-baseweb="select"] input { color: #F2F5F9; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def read_parquet(filename):
    path = ANALYTICS / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing analytics file: {path}")
    return pd.read_parquet(path)

@st.cache_data(show_spinner=False)
def read_metadata():
    with MODEL.open("r", encoding="utf-8") as handle:
        return json.load(handle)

@st.cache_data(show_spinner=False)
def read_cleaned_sample(limit=2500):
    sample_path = (
        ROOT / "data" / "processed" / "cleaned_sample.parquet"
    )

    full_path = (
        ROOT / "data" / "processed"
        / "firewatch_cleaned_file.parquet"
    )

    # Prefer the small deployment-ready sample.
    if sample_path.is_file():
        return pd.read_parquet(sample_path).head(limit)

    # Local fallback when the full cleaned dataset exists.
    if full_path.is_file():
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(full_path)
        batch = next(parquet.iter_batches(batch_size=limit))
        return batch.to_pandas().head(limit)

    return pd.DataFrame()

try:
    monthly = read_parquet("monthly_statistics.parquet").sort_values("month").copy()
    categories = read_parquet("frp_categories.parquet").copy()
    clusters = read_parquet("cluster_statistics.parquet").sort_values("cluster").copy()
    hotspots = read_parquet("geographic_hotspots.parquet").copy()
    daynight = read_parquet("daynight_statistics.parquet").copy()
    confidence = read_parquet("confidence_statistics.parquet").copy()
    metadata = read_metadata()
except Exception as exc:
    st.error(f"Could not load FireWatch project data: {exc}")
    st.info("Run the Spark analytics and K-Means export scripts first, then refresh this page.")
    st.stop()

MONTHS = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
monthly["month_name"] = monthly["month"].map(MONTHS)
hotspots["latitude"] = hotspots["lat_grid"] + .5
hotspots["longitude"] = hotspots["lon_grid"] + .5
clusters["cluster_label"] = "Cluster " + clusters["cluster"].astype(str)
total = int(clusters["detections"].sum())
k = int(metadata["k"])
score = float(metadata["silhouette_score"])
peak = monthly.loc[monthly["total_detections"].idxmax()]

COLORS = ["#FFB15B", "#FF8A3D", "#FF514F", "#B82D43", "#FFE1A0"]

def theme(fig, height=400, title=None):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="Arial"),
        title=dict(text=title, font=dict(size=16, color=TEXT), x=.025, y=.96),
        margin=dict(l=24, r=20, t=60 if title else 28, b=25), height=height,
        colorway=COLORS,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
        hoverlabel=dict(bgcolor="#202A37", font_color=TEXT),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, color=MUTED, linecolor=EDGE)
    fig.update_yaxes(gridcolor="#293341", zeroline=False, color=MUTED)
    return fig


def header(section, title, description):
    st.markdown(f'<div class="fire-eyebrow">{section.upper()}</div><div class="fire-title">{title}</div><div class="fire-sub">{description}</div>', unsafe_allow_html=True)


def plot(fig):
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def monthly_fig():
    fig = px.bar(monthly, x="month_name", y="total_detections", color="total_detections",
                 color_continuous_scale=["#60392D", "#FF9C47", "#FF514F"],
                 labels={"month_name":"Month", "total_detections":"Detections"})
    fig.update_layout(coloraxis_showscale=False)
    fig.update_traces(marker_line_width=0, hovertemplate="%{x}: %{y:,} detections<extra></extra>")
    return theme(fig, 365, "Monthly satellite detections")


def frp_fig():
    fig = px.line(monthly, x="month_name", y="avg_frp", markers=True,
                  labels={"month_name":"Month", "avg_frp":"Average FRP (MW)"})
    fig.update_traces(line=dict(color=ORANGE, width=3), marker=dict(color=GOLD, size=8),
                      hovertemplate="%{x}: %{y:.2f} MW<extra></extra>")
    return theme(fig, 365, "Average FRP by month")


def category_fig():
    order = [x for x in ["Low", "Moderate", "High"] if x in categories["frp_category"].tolist()]
    fig = px.pie(categories, names="frp_category", values="detections", hole=.66,
                 category_orders={"frp_category":order},
                 color="frp_category", color_discrete_map={"Low":"#FFB15B", "Moderate":"#FF8A3D", "High":"#FF514F"})
    fig.update_traces(textinfo="percent", textfont=dict(color=TEXT, size=13),
                      marker=dict(line=dict(color=PANEL, width=3)))
    fig.update_layout(annotations=[dict(text="FRP", x=.5, y=.5, showarrow=False, font=dict(size=22, color=TEXT))])
    return theme(fig, 365, "FRP detection categories")


def cluster_fig():
    fig = px.bar(clusters, x="cluster_label", y="detections", color="avg_frp",
                 color_continuous_scale=["#FFB15B", "#FF8A3D", "#D94B42"],
                 labels={"cluster_label":"Cluster", "detections":"Detections", "avg_frp":"Avg FRP (MW)"})
    fig.update_traces(marker_line_width=0, hovertemplate="%{x}: %{y:,} detections<extra></extra>")
    fig.update_layout(coloraxis_colorbar=dict(title="Avg FRP", tickfont=dict(color=MUTED)))
    return theme(fig, 385, "Detection counts by geographic cluster")


# Satellite view uses MapLibre/WebGL; the 2D map below is an SVG-based,
# webgl-independent fallback that intentionally does NOT imply a real satellite basemap.
INTERACTIVE_STYLES = ["satellite", "satellite-streets", "carto-darkmatter", "open-street-map"]
FIRE_SCALE = ["#FFE4A0", "#FFBD63", "#FF8234", "#D83630"]
SCORE_SCALE = ["#376F99", "#F0CF86", "#FA9D4D", "#E34237"]
SEA_COLOR = "#103B55"
LAND_COLOR = "#B98D60"


def style_2d_geo(fig, height=590, score=False):
    """Plotly Geo uses SVG; this rendering does not need WebGL2."""
    if map_extent == "Mainland & Tasmania":
        # Enlarged Australian mainland/Tasmania viewport. Some offshore
        # territories, e.g. Macquarie Island, lie outside this crop.
        lon_range, lat_range = [110, 155], [-46, -9]
    else:
        lon_range, lat_range = [108, 165], [-57, -8]

    fig.update_geos(
        projection_type="mercator",
        showland=True, landcolor=LAND_COLOR,
        showocean=True, oceancolor=SEA_COLOR,
        showlakes=True, lakecolor="#276784",
        showrivers=True, rivercolor="#39788E",
        showcoastlines=True, coastlinecolor="#F1D0A0",
        coastlinewidth=1.1,
        showcountries=True, countrycolor="#947550", countrywidth=0.6,
        lonaxis_range=lon_range,
        lataxis_range=lat_range,
        bgcolor=SEA_COLOR,
        showframe=False,
        resolution=50,
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PANEL,
        font=dict(color=TEXT, family="Arial"),
        margin=dict(l=5, r=5, t=8, b=8),
        height=height,
        coloraxis_colorbar=dict(
            title="Score (%)" if score else "Avg FRP (MW)",
            thickness=13, len=0.82, tickfont=dict(color=MUTED),
        ),
        legend=dict(bgcolor="rgba(16,29,39,.85)", x=0.01, y=0.02),
        geo=dict(domain=dict(x=[0, 0.97], y=[0, 1])),
    )
    return fig


def map_fig(grid, show_centers=True, map_style="satellite"):
    """Show spatial analysis using a safe 2D map or MapLibre satellite map."""
    hover = {
        "detections": ":,",
        "avg_frp": ":.2f",
        "max_frp": ":.2f",
        "lat_grid": True,
        "lon_grid": True,
        "latitude": False,
        "longitude": False,
    }
    labels = {"avg_frp": "Average FRP (MW)"}
    if use_2d_maps:
        fig = px.scatter_geo(
            grid, lat="latitude", lon="longitude", size="detections",
            color="avg_frp", size_max=20,
            color_continuous_scale=FIRE_SCALE,
            hover_data=hover, labels=labels,
        )
        fig.update_traces(
            marker=dict(opacity=0.88, line=dict(color="#553723", width=0.45)),
            selector=dict(type="scattergeo"),
        )
        if show_centers:
            centers = metadata.get("cluster_centers", [])
            if centers:
                fig.add_trace(go.Scattergeo(
                    lat=[c["latitude"] for c in centers],
                    lon=[c["longitude"] for c in centers],
                    mode="markers+text",
                    text=[f"C{c['cluster']}" for c in centers],
                    textposition="top center",
                    textfont=dict(color="#F4FFFF", size=12),
                    marker=dict(size=13, color="#27D9E5", line=dict(width=1, color="#061B26")),
                    name="K-Means centers",
                    hovertext=[
                        f"Cluster {c['cluster']}<br>Lat: {c['latitude']:.4f}<br>Lon: {c['longitude']:.4f}"
                        for c in centers
                    ],
                    hoverinfo="text",
                ))
        return style_2d_geo(fig, height=580)

    fig = px.scatter_map(
        grid, lat="latitude", lon="longitude", size="detections",
        color="avg_frp", size_max=22, zoom=3.15,
        center={"lat": -25.5, "lon": 134.5},
        map_style=map_style,
        color_continuous_scale=FIRE_SCALE,
        hover_data=hover, labels=labels,
    )
    fig.update_traces(marker=dict(opacity=0.86), selector=dict(type="scattermap"))
    if show_centers:
        centers = metadata.get("cluster_centers", [])
        if centers:
            fig.add_trace(go.Scattermap(
                lat=[c["latitude"] for c in centers],
                lon=[c["longitude"] for c in centers],
                mode="markers+text",
                text=[f"C{c['cluster']}" for c in centers],
                textposition="top center",
                textfont=dict(color="#FFFFFF", size=14),
                marker=dict(size=19, color="#27D9E5", opacity=1),
                name="K-Means centers",
                hovertext=[
                    f"Cluster {c['cluster']}<br>Lat: {c['latitude']:.4f}<br>Lon: {c['longitude']:.4f}"
                    for c in centers
                ],
                hoverinfo="text",
            ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=PANEL, font=dict(color=TEXT),
        height=580, margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(title="Avg FRP (MW)", tickfont=dict(color=MUTED)),
        legend=dict(bgcolor="rgba(20,27,38,.85)", x=.01, y=.01),
        map=dict(bearing=0, pitch=0),
    )
    return fig

with st.sidebar:
    st.markdown('<div class="fire-brand">🔥 FireWatch <span>AI</span></div>', unsafe_allow_html=True)
    st.caption("SATELLITE INTELLIGENCE PLATFORM")
    st.markdown('<span class="fire-tag">● AUSTRALIA · 2024</span>', unsafe_allow_html=True)
    st.divider()
    page = st.radio("NAVIGATION", ["🏠 Overview", "📈 Temporal Analytics", "🗺️ Geographic Intelligence", "🧠 ML Intelligence", "🔥 Hotspot Prediction", "🗃️ Data Explorer"], label_visibility="collapsed")
    use_2d_maps = st.toggle(
        "2D Compatibility Maps",
        value=True,
        help="SVG-based map for devices without WebGL2. Disable for interactive satellite imagery."
    )
    map_extent = "Mainland & Tasmania"
    if use_2d_maps:
        map_extent = st.selectbox(
            "2D map coverage",
            ["Mainland & Tasmania", "All Australian territories"],
            index=0,
            help="The wider view also includes remote Australian island cells."
        )
    else:
        st.caption("Satellite basemaps need WebGL2 and internet access.")
    st.divider()
    st.caption("DATA SOURCE  ·  NASA FIRMS VIIRS S-NPP")
    st.caption("ENGINE  ·  APACHE SPARK + MLLIB")
    st.caption("Satellite thermal detections ≠ confirmed wildfire incidents")

if page == "🏠 Overview":
    header("Command Center / Overview", "Wildfire activity, at a glance.", "Australia 2024 · NASA FIRMS VIIRS S-NPP · Historical satellite thermal detections")
    a,b,c,d = st.columns(4)
    a.metric("THERMAL DETECTIONS", f"{total:,}")
    b.metric("GEOGRAPHIC CLUSTERS", str(k))
    c.metric("SILHOUETTE SCORE", f"{score:.4f}")
    d.metric("PEAK DETECTION MONTH", MONTHS.get(int(peak["month"]), str(peak["month"])))
    st.write("")
    st.markdown("### 🗺️ Detection hotspot map")
    top_grid = hotspots[hotspots["detections"] >= hotspots["detections"].quantile(.35)]
    plot(map_fig(top_grid, True))
    st.caption("Circle size = detections per 1° cell · Color = mean FRP (MW) · Cyan = K-Means centers. 2D mode uses stylized terrain colors, not satellite imagery.")
    left,right = st.columns(2)
    with left: plot(monthly_fig())
    with right: plot(frp_fig())
    st.markdown('<div class="fire-panel"><strong>What this platform measures</strong><p>These are satellite thermal detections, not confirmed individual wildfires. Detection density and FRP vary with satellite observation conditions and should not be interpreted as incident counts or validated fire severity.</p></div>', unsafe_allow_html=True)

elif page == "📈 Temporal Analytics":
    header("Analytics / Time Series", "When did thermal activity peak?", "Explore detection counts, radiative power, and observation categories across 2024.")
    a,b,c = st.columns(3)
    a.metric("PEAK MONTH", MONTHS.get(int(peak["month"]), "—"))
    b.metric("PEAK MONTH DETECTIONS", f"{int(peak['total_detections']):,}")
    highest_frp = monthly.loc[monthly["avg_frp"].idxmax()]
    c.metric("HIGHEST MONTHLY AVG FRP", f"{float(highest_frp['avg_frp']):.2f} MW")
    left,right = st.columns(2)
    with left: plot(monthly_fig())
    with right: plot(frp_fig())
    left,right = st.columns(2)
    with left: plot(category_fig())
    with right:
        st.markdown("#### Project FRP bands")
        st.markdown('<div class="fire-panel"><strong>Low</strong><p>FRP below 10 MW</p><br><strong>Moderate</strong><p>FRP from 10 MW to below 50 MW</p><br><strong>High</strong><p>FRP of 50 MW or above</p></div>', unsafe_allow_html=True)
        st.caption("These bands were defined for this project; they are not official wildfire severity classes.")
    st.markdown("### Monthly summary")
    st.dataframe(monthly, width="stretch", hide_index=True)
    st.markdown("### Observation breakdowns")
    left,right = st.columns(2)
    with left:
        st.markdown("**Day / Night**")
        st.dataframe(daynight, width="stretch", hide_index=True)
    with right:
        st.markdown("**Satellite confidence categories**")
        st.dataframe(confidence, width="stretch", hide_index=True)

elif page == "🗺️ Geographic Intelligence":
    header("Analytics / Geographic", "Explore Australia's hotspots.", "Interactive spatial aggregation of historical VIIRS thermal detections.")
    a,b,c = st.columns([2,1,1])
    with a:
        minimum = st.slider("Minimum detections per 1° grid cell", min_value=0, max_value=int(hotspots["detections"].max()), value=0, step=100)
    with b:
        centers = st.checkbox("Show K-Means centers", value=True)
    with c:
        map_style = st.selectbox(
            "Interactive basemap", INTERACTIVE_STYLES, index=0,
            disabled=use_2d_maps,
            help="Available when 2D Compatibility Maps is switched off."
        )
    grid = hotspots[hotspots["detections"] >= minimum].copy()
    if grid.empty:
        st.warning("No grid cells match the current filter.")
    else:
        x,y,z = st.columns(3)
        x.metric("VISIBLE GRID CELLS", f"{len(grid):,}")
        y.metric("DETECTIONS IN VISIBLE CELLS", f"{int(grid['detections'].sum()):,}")
        z.metric("MAX CELL DETECTIONS", f"{int(grid['detections'].max()):,}")
        plot(map_fig(grid, centers, map_style))
        st.caption("1° × 1° geographic cells; cluster centers are model centroids, not exact fire locations. The 2D map is stylized; satellite imagery requires interactive mode. Filters do not retrain the model.")
        st.markdown("### Highest-detection grid cells")
        st.dataframe(grid.sort_values("detections", ascending=False).head(20)[["lat_grid", "lon_grid", "detections", "avg_frp", "max_frp"]], width="stretch", hide_index=True)

elif page == "🧠 ML Intelligence":
    header("Machine Learning / Spark MLlib", "Geographic clustering intelligence.", "K-Means clustering of latitude and longitude across the cleaned satellite detections.")
    a,b,c = st.columns(3)
    a.metric("ALGORITHM", "K-Means")
    b.metric("CLUSTERS", str(k))
    c.metric("SILHOUETTE", f"{score:.4f}")
    plot(cluster_fig())
    left,right = st.columns([1.3,1])
    with left:
        st.markdown("### Cluster centers")
        plot(map_fig(hotspots[hotspots["detections"] >= hotspots["detections"].quantile(.55)], True))
    with right:
        st.markdown("### Cluster statistics")
        st.dataframe(clusters[["cluster", "detections", "center_lat", "center_lon", "avg_frp"]], width="stretch", hide_index=True)
        st.markdown('<div class="fire-panel"><strong>Model interpretation</strong><p>K-Means identifies broad geographic groups of detections. It does not predict future fires or classify confirmed wildfire events.</p></div>', unsafe_allow_html=True)
        st.caption("Distance metric: Euclidean distance on latitude/longitude degrees (baseline; not geodesic distance).")
    with st.expander("View model metadata"):
        st.json(metadata)

elif page == "🔥 Hotspot Prediction":
    from datetime import date

    header(
        "Machine Learning / Geographic Prediction",
        "Explore geographic thermal-detection scores.",
        "Spark MLlib Random Forest · 1° Australia grid · Historical test and experimental scenario"
    )
    metrics_path = ROOT / "models" / "geographic_rf_metrics.json"
    test_path = ROOT / "data" / "processed" / "geographic_predictions_2024_test.parquet"
    jan_path = ROOT / "data" / "processed" / "geographic_predictions_2025_01.parquet"

    @st.cache_data(show_spinner=False)
    def load_prediction_data(path):
        return pd.read_parquet(path)

    st.markdown("### Choose prediction mode")
    view_mode = st.radio(
        "Prediction mode",
        ["2024 held-out test (Nov–Dec)", "January 2025 experimental scenario"],
        horizontal=True,
        label_visibility="collapsed",
        key="prediction_view_mode"
    )
    test_mode = view_mode.startswith("2024")
    data_path = test_path if test_mode else jan_path
    if not data_path.is_file():
        st.warning(f"Prediction export is missing: {data_path.name}")
        st.code("python src/12_generate_geographic_predictions.py", language="powershell")
        st.info("Generate the small offline prediction files first, then refresh the page. The dashboard does not train Spark models.")
        st.stop()

    try:
        source_forecast = load_prediction_data(str(data_path)).copy()
    except Exception as exc:
        st.error(f"Cannot open prediction data: {exc}")
        st.stop()

    required = {
        "lat_grid", "lon_grid", "month", "forecast_year", "prediction",
        "probability_detection", "probability_no_detection",
        "previous_1_month_detections", "previous_2_month_detections",
        "previous_3_month_detections", "actual_detection", "actual_detections"
    }
    missing = required - set(source_forecast.columns)
    if missing:
        st.error(f"Missing forecast columns: {sorted(missing)}")
        st.stop()
    not_nullable = sorted(required - {"actual_detection", "actual_detections"})
    if source_forecast.empty or source_forecast[not_nullable].isna().any().any():
        st.error("Forecast file is empty or has missing required inputs or outputs.")
        st.stop()
    if source_forecast[["lat_grid", "lon_grid", "month", "forecast_year"]].duplicated().any():
        st.error("Duplicate grid/month combinations in prediction data.")
        st.stop()
    if (not source_forecast["probability_detection"].between(0, 1).all()
            or not source_forecast["probability_no_detection"].between(0, 1).all()):
        st.error("Prediction scores must be between 0 and 1.")
        st.stop()
    if not source_forecast["prediction"].isin([0, 1]).all():
        st.error("Unexpected predicted classes.")
        st.stop()

    if test_mode:
        if not ((source_forecast["forecast_year"] == 2024)
                & source_forecast["month"].isin([11, 12])).all():
            st.error("Expected November/December 2024 held-out predictions.")
            st.stop()
        if source_forecast["actual_detection"].isna().any():
            st.error("Held-out test file is missing true labels.")
            st.stop()
        st.success(
            "Held-out historical test: 2024 observations were used to check model predictions. "
            "Results show satellite thermal detections, not confirmed fires."
        )
        month_to_show = st.selectbox(
            "Test month", [11, 12],
            format_func=lambda m: f"{MONTHS[m]} 2024",
            key="prediction_test_month"
        )
        forecast = source_forecast[source_forecast["month"] == month_to_show].copy()
        selected_year, selected_month = 2024, month_to_show
        overall_y = source_forecast["actual_detection"].astype(int)
        overall_rf = source_forecast["prediction"].astype(int)
        overall_baseline = (source_forecast["previous_1_month_detections"] > 0).astype(int)
        full_rf_accuracy = float((overall_y == overall_rf).mean())
        full_baseline_accuracy = float((overall_y == overall_baseline).mean())
        st.caption(
            f"Complete Nov–Dec holdout: Random Forest accuracy {full_rf_accuracy:.2%} "
            f"vs persistence baseline {full_baseline_accuracy:.2%}; "
            "the model was fitted on Apr–Sep 2024 and evaluated chronologically."
        )
    else:
        if not ((source_forecast["forecast_year"] == 2025)
                & (source_forecast["month"] == 1)).all():
            st.error("Expected January 2025 experimental scenario.")
            st.stop()
        forecast = source_forecast.copy()
        selected_year, selected_month = 2025, 1
        st.warning(
            "Unverified retrospective scenario, NOT a current or live fire forecast. "
            "The model used October–December 2024 detection history to score January 2025. "
            "January is outside the training target months (Apr–Sep 2024), "
            "and actual January 2025 outcomes were not collected or checked here. "
            "Scores are uncalibrated model outputs, not validated real-world fire probabilities."
        )

    if len(forecast) != 822 or forecast[["lat_grid", "lon_grid"]].duplicated().any():
        st.error(f"Expected exactly 822 unique geographic cells for this period, got {len(forecast)}.")
        st.stop()

    forecast["latitude"] = forecast["lat_grid"] + .5
    forecast["longitude"] = forecast["lon_grid"] + .5
    forecast["probability_pct"] = forecast["probability_detection"] * 100
    forecast["cell_label"] = forecast.apply(
        lambda r: f"Grid ({int(r['lat_grid'])}, {int(r['lon_grid'])})", axis=1
    )

    a, b, c, d = st.columns(4)
    a.metric("GEOGRAPHIC GRID CELLS", f"{len(forecast):,}")
    b.metric("PREDICTED DETECTION", f"{int((forecast['prediction'] == 1).sum()):,}")
    if test_mode:
        c.metric("OBSERVED DETECTION", f"{int((forecast['actual_detection'] == 1).sum()):,}")
        month_acc = (forecast["prediction"].astype(int) == forecast["actual_detection"].astype(int)).mean()
        d.metric("SELECTED MONTH ACCURACY", f"{month_acc:.2%}")
    else:
        c.metric("PREDICTED NO DETECTION", f"{int((forecast['prediction'] == 0).sum()):,}")
        d.metric("MEAN MODEL SCORE", f"{forecast['probability_detection'].mean():.1%}")

    st.markdown("### 🗺️ Geographic model-score map")
    filter_col, style_col = st.columns([2, 1])
    with filter_col:
        threshold = st.slider("Minimum model score (%)", 0, 100, 0, 5, key="prediction_score_threshold")
    with style_col:
        prediction_map_style = st.selectbox(
            "Interactive basemap", INTERACTIVE_STYLES,
            index=0, key="prediction_map_style", disabled=use_2d_maps,
            help="Available when 2D Compatibility Maps is switched off."
        )
    visible = forecast.loc[forecast["probability_pct"] >= threshold].copy()
    st.caption(
        f"Showing {len(visible):,} of {len(forecast):,} cells · "
        "Markers are grid-cell centers, not fire locations · Marker color = model score"
    )
    if visible.empty:
        st.info("No cells satisfy the selected score threshold.")
    else:
        score_hover = {
            "probability_pct": ":.1f",
            "previous_1_month_detections": ":,.0f",
            "previous_2_month_detections": ":,.0f",
            "previous_3_month_detections": ":,.0f",
            "latitude": False,
            "longitude": False,
        }
        if use_2d_maps:
            prediction_map = px.scatter_geo(
                visible, lat="latitude", lon="longitude",
                color="probability_pct", color_continuous_scale=SCORE_SCALE,
                range_color=[0, 100], hover_name="cell_label",
                hover_data=score_hover,
                labels={"probability_pct": "Model score (%)"},
            )
            prediction_map.update_traces(
                marker=dict(size=8.5, opacity=.92,
                            line=dict(width=.4, color="#4C311F")),
                selector=dict(type="scattergeo"),
            )
            style_2d_geo(prediction_map, height=580, score=True)
        else:
            prediction_map = px.scatter_map(
                visible, lat="latitude", lon="longitude",
                color="probability_pct", color_continuous_scale=SCORE_SCALE,
                range_color=[0, 100], hover_name="cell_label",
                hover_data=score_hover,
                labels={"probability_pct": "Model score (%)"},
                zoom=3.15, center={"lat": -25.5, "lon": 134.5},
                map_style=prediction_map_style,
            )
            prediction_map.update_traces(marker=dict(size=10, opacity=.87))
            prediction_map.update_layout(
                template="plotly_dark", paper_bgcolor=PANEL,
                font=dict(color=TEXT), height=580,
                margin=dict(l=0, r=0, t=0, b=0),
                coloraxis_colorbar=dict(title="Score (%)", tickfont=dict(color=MUTED)),
            )
        plot(prediction_map)

    st.markdown("### 📍 Inspect a geographic grid cell")
    selected_cell = st.selectbox(
        "Choose a geographic grid cell",
        options=forecast.sort_values(["lat_grid", "lon_grid"])["cell_label"].tolist(),
        index=0,
        key="prediction_selected_cell",
    )
    chosen = forecast.loc[forecast["cell_label"] == selected_cell].iloc[0]
    left, right = st.columns([1, 1.4])
    with left:
        st.metric("MODEL DETECTION SCORE", f"{chosen['probability_detection']:.1%}")
        st.metric("PREDICTED CLASS", "Detection" if int(chosen["prediction"]) else "No detection")
        if test_mode:
            observed_text = "Detection" if int(chosen["actual_detection"]) else "No detection"
            st.metric("OBSERVED CLASS", observed_text)
            st.caption(f"Actual detected records in month: {int(chosen['actual_detections']):,}")
        st.caption(
            f"Grid bounds: {int(chosen['lat_grid'])}° to {int(chosen['lat_grid']) + 1}° latitude; "
            f"{int(chosen['lon_grid'])}° to {int(chosen['lon_grid']) + 1}° longitude"
        )

    def historical_label(months_before):
        # Convert a 1-based year/month to an offset without extra dependencies.
        index = selected_year * 12 + (selected_month - 1) - months_before
        year, month_zero = divmod(index, 12)
        return date(year, month_zero + 1, 1).strftime("%b %Y")

    with right:
        history = pd.DataFrame({
            "Month": [historical_label(3), historical_label(2), historical_label(1)],
            "Detections": [
                int(chosen["previous_3_month_detections"]),
                int(chosen["previous_2_month_detections"]),
                int(chosen["previous_1_month_detections"]),
            ],
        })
        hist_fig = px.bar(history, x="Month", y="Detections", text="Detections")
        hist_fig.update_traces(marker_color=ORANGE, textposition="outside", cliponaxis=False)
        plot(theme(hist_fig, 310, "Detection history used for this prediction"))

    st.markdown("### Highest-scoring geographic cells")
    table_cols = [
        "lat_grid", "lon_grid", "probability_pct", "prediction",
        "previous_1_month_detections", "previous_2_month_detections",
        "previous_3_month_detections",
    ]
    if test_mode:
        table_cols += ["actual_detection", "actual_detections"]
    top_forecast = forecast.nlargest(20, "probability_detection")[table_cols].rename(
        columns={"probability_pct": "Model score (%)"}
    )
    st.dataframe(top_forecast, width="stretch", hide_index=True)
    st.download_button(
        "Download displayed month predictions (CSV)",
        forecast.drop(columns=["latitude", "longitude", "cell_label", "probability_pct"])
        .to_csv(index=False).encode("utf-8"),
        file_name=f"firewatch_predictions_{selected_year}_{selected_month:02d}.csv",
        mime="text/csv",
    )
    with st.expander("Method, comparison, and limitations"):
        st.markdown(
            "**Model:** 100-tree Random Forest trained in Spark MLlib using grid latitude, "
            "grid longitude, target month, and previous one, two and three months' detection counts. "
            "**Temporal split:** Apr–Sep 2024 training, October validation, Nov–Dec testing. "
            "**Baseline:** predict a detection when the previous month had at least one detection. "
            "**Limitations:** one data year; retrospective evaluation only; no external-year validation. "
            "An absence of satellite detections does not establish an absence of fires. "
            "The model score is not a calibrated fire probability. "
            "The January 2025 scenario extrapolates to an unseen target month and is not validated."
        )
        if metrics_path.is_file():
            with metrics_path.open("r", encoding="utf-8") as handle:
                st.json(json.load(handle))

else:
    header("Data / Explorer", "Explore the processed analytics.", "Inspect aggregated datasets, filter rows, and export selected tables as CSV.")
    available = {
        "Monthly statistics": monthly,
        "FRP categories": categories,
        "Geographic hotspots": hotspots,
        "K-Means cluster statistics": clusters,
        "Day / Night statistics": daynight,
        "Confidence statistics": confidence,
    }
    dataset = st.selectbox("Dataset", list(available))
    frame = available[dataset]
    limit = st.slider("Rows to display", 10, 1000, 100, 10)
    st.metric("ROWS IN DATASET", f"{len(frame):,}")
    st.dataframe(frame.head(limit), width="stretch", hide_index=True)
    st.download_button("Download dataset as CSV", frame.to_csv(index=False).encode("utf-8"), file_name=dataset.lower().replace(" ", "_").replace("/", "_") + ".csv", mime="text/csv")
    with st.expander("Preview cleaned source detections (first 2,500 rows)"):
        if st.button("Load sample of cleaned detections"):
            sample = read_cleaned_sample()
            if sample.empty:
                st.warning("Cleaned preview not found. Add data/processed/cleaned_sample.parquet to the deployed repository.")
            else:
                st.dataframe(sample, width="stretch", hide_index=True)
                st.caption("Preview only; not a random sample and not the full dataset.")

st.divider()
st.caption("FIREWATCH AI  ·  NASA FIRMS VIIRS S-NPP  ·  AUSTRALIA 2024  ·  APACHE SPARK + SPARK MLLIB  ·  HISTORICAL ANALYTICS + EXPERIMENTAL FORECAST")
