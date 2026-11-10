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
    path = ROOT / "data" / "processed" / "firewatch_cleaned_file.parquet"
    if not path.is_file():
        return pd.DataFrame()
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    chunks = []
    remaining = limit
    for batch in parquet.iter_batches(batch_size=min(remaining, 2500)):
        frame = batch.to_pandas()
        chunks.append(frame.head(remaining))
        remaining -= len(chunks[-1])
        if remaining <= 0:
            break
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

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
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


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


def map_fig(grid, show_centers=True, map_style="carto-darkmatter"):
    fig = px.scatter_map(grid, lat="latitude", lon="longitude", size="detections", color="avg_frp",
                         size_max=22, zoom=3, center={"lat":-25.5,"lon":134.5},
                         map_style=map_style,
                         color_continuous_scale=["#FFE0A0", "#FF8A3D", "#E23A3A"],
                         hover_data={"detections":":,", "avg_frp":":.2f", "max_frp":":.2f",
                                     "latitude":False,"longitude":False,"lat_grid":True,"lon_grid":True},
                         labels={"avg_frp":"Avg FRP (MW)"})
    fig.update_traces(marker=dict(opacity=.80), selector=dict(type="scattermap"))
    if show_centers:
        centers = metadata.get("cluster_centers", [])
        if centers:
            fig.add_trace(go.Scattermap(
                lat=[c["latitude"] for c in centers], lon=[c["longitude"] for c in centers],
                mode="markers+text", text=[f"C{c['cluster']}" for c in centers],
                textposition="top center", textfont=dict(color="#FFFFFF", size=14),
                marker=dict(size=19, color="#35D6E8", opacity=1),
                name="K-Means centers", hovertext=[f"Cluster {c['cluster']}<br>Lat: {c['latitude']:.4f}<br>Lon: {c['longitude']:.4f}" for c in centers],
                hoverinfo="text"))
    fig.update_layout(template="plotly_dark", paper_bgcolor=PANEL, font=dict(color=TEXT),
                      height=610, margin=dict(l=0,r=0,t=0,b=0),
                      coloraxis_colorbar=dict(title="Avg FRP", tickfont=dict(color=MUTED)),
                      legend=dict(bgcolor="rgba(20,27,38,.85)", x=.01, y=.01),
                      map=dict(bearing=0, pitch=0))
    return fig

with st.sidebar:
    st.markdown('<div class="fire-brand">🔥 FireWatch <span>AI</span></div>', unsafe_allow_html=True)
    st.caption("SATELLITE INTELLIGENCE PLATFORM")
    st.markdown('<span class="fire-tag">● AUSTRALIA · 2024</span>', unsafe_allow_html=True)
    st.divider()
    page = st.radio("NAVIGATION", ["🏠 Overview", "📈 Temporal Analytics", "🗺️ Geographic Intelligence", "🧠 ML Intelligence", "🗃️ Data Explorer"], label_visibility="collapsed")
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
    st.caption("Circle size = number of detections per 1° grid cell · Color = average FRP · Cyan markers = K-Means geographic centers")
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
    st.dataframe(monthly, use_container_width=True, hide_index=True)
    st.markdown("### Observation breakdowns")
    left,right = st.columns(2)
    with left:
        st.markdown("**Day / Night**")
        st.dataframe(daynight, use_container_width=True, hide_index=True)
    with right:
        st.markdown("**Satellite confidence categories**")
        st.dataframe(confidence, use_container_width=True, hide_index=True)

elif page == "🗺️ Geographic Intelligence":
    header("Analytics / Geographic", "Explore Australia's hotspots.", "Interactive spatial aggregation of historical VIIRS thermal detections.")
    a,b,c = st.columns([2,1,1])
    with a:
        minimum = st.slider("Minimum detections per 1° grid cell", min_value=0, max_value=int(hotspots["detections"].max()), value=0, step=100)
    with b:
        centers = st.checkbox("Show K-Means centers", value=True)
    with c:
        map_style = st.selectbox("Basemap", ["carto-darkmatter", "open-street-map"], index=0)
    grid = hotspots[hotspots["detections"] >= minimum].copy()
    if grid.empty:
        st.warning("No grid cells match the current filter.")
    else:
        x,y,z = st.columns(3)
        x.metric("VISIBLE GRID CELLS", f"{len(grid):,}")
        y.metric("DETECTIONS IN VISIBLE CELLS", f"{int(grid['detections'].sum()):,}")
        z.metric("MAX CELL DETECTIONS", f"{int(grid['detections'].max()):,}")
        plot(map_fig(grid, centers, map_style))
        st.caption("Grid cells are 1° latitude × 1° longitude. Geographic centers are cluster centroids, not actual wildfire locations. Filtering cells does not retrain the model.")
        st.markdown("### Highest-detection grid cells")
        st.dataframe(grid.sort_values("detections", ascending=False).head(20)[["lat_grid", "lon_grid", "detections", "avg_frp", "max_frp"]], use_container_width=True, hide_index=True)

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
        st.dataframe(clusters[["cluster", "detections", "center_lat", "center_lon", "avg_frp"]], use_container_width=True, hide_index=True)
        st.markdown('<div class="fire-panel"><strong>Model interpretation</strong><p>K-Means identifies broad geographic groups of detections. It does not predict future fires or classify confirmed wildfire events.</p></div>', unsafe_allow_html=True)
        st.caption("Distance metric: Euclidean distance on latitude/longitude degrees (baseline; not geodesic distance).")
    with st.expander("View model metadata"):
        st.json(metadata)

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
    st.dataframe(frame.head(limit), use_container_width=True, hide_index=True)
    st.download_button("Download dataset as CSV", frame.to_csv(index=False).encode("utf-8"), file_name=dataset.lower().replace(" ", "_").replace("/", "_") + ".csv", mime="text/csv")
    with st.expander("Preview cleaned source detections (first 2,500 rows)"):
        if st.button("Load sample of cleaned detections"):
            sample = read_cleaned_sample()
            if sample.empty:
                st.warning("Cleaned Parquet file was not found.")
            else:
                st.dataframe(sample, use_container_width=True, hide_index=True)
                st.caption("Preview only; not a random sample and not the full dataset.")

st.divider()
st.caption("FIREWATCH AI  ·  NASA FIRMS VIIRS S-NPP  ·  AUSTRALIA 2024  ·  APACHE SPARK + SPARK MLLIB  ·  HISTORICAL ANALYTICS")
