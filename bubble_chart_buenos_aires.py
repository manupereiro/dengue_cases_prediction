import geopandas as gpd
import pandas as pd
import plotly.express as px

# === CARGAR DATOS ===

# 1. Leer shapefile de comunas de CABA
gdf = gpd.read_file("data/comunas/comunas.shp").explode(index_parts=False)
gdf["comuna"] = gdf["comuna"].astype(str).str.strip().str.zfill(2)

# 2. Leer CSV con datos de dengue
df = pd.read_csv("data/dengue_unified_dataset.csv")
df = df[df["year"] == 2025]
# === DATOS CABA ===

df_caba = df[df["province"] == "CABA"].copy()

# Condiciones favorables
df_caba["dengue_favorable"] = (
    (df_caba["temp_2weeks_avg"] > 25) &
    (df_caba["humd_2weeks_avg"] > 60) &
    (df_caba["prec_2weeks"] > 20)
).astype(int)

# Agrupar por comuna
agg_caba = df_caba.groupby("departament_name").agg({
    "dengue_favorable": "sum",
    "target_cases": "mean"
}).reset_index()

agg_caba.rename(columns={"departament_name": "comuna"}, inplace=True)
agg_caba["comuna"] = agg_caba["comuna"].str.upper().str.replace("COMUNA ", "").str.zfill(2)

# Unir con geometría
merged = gdf.merge(agg_caba, on="comuna", how="inner")

# Centroides para burbujas
merged = merged.to_crs(epsg=3857)
merged["centroid"] = merged.geometry.centroid
merged["lon"] = merged.centroid.to_crs(epsg=4326).x
merged["lat"] = merged.centroid.to_crs(epsg=4326).y

# === DATOS GBA ===

df_gba = df[df["province"] == "Buenos Aires"].copy()

df_gba["dengue_favorable"] = (
    (df_gba["temp_2weeks_avg"] > 25) &
    (df_gba["humd_2weeks_avg"] > 60) &
    (df_gba["prec_2weeks"] > 20)
).astype(int)

agg_gba = df_gba.groupby("departament_name").agg({
    "dengue_favorable": "sum",
    "target_cases": "mean"
}).reset_index()

# Coordenadas aproximadas por municipio del GBA
coords = {
    "La Matanza": (-34.749, -58.585),
    "Avellaneda": (-34.660, -58.366),
    "Lanús": (-34.708, -58.397),
    "Lomas de Zamora": (-34.762, -58.406),
    "Quilmes": (-34.721, -58.254),
    "Florencio Varela": (-34.815, -58.275),
    "Berazategui": (-34.764, -58.213),
    "Almirante Brown": (-34.802, -58.393),
    "San Isidro": (-34.473, -58.528),
    "Vicente López": (-34.531, -58.479),
    "Tres de Febrero": (-34.611, -58.563),
    "Morón": (-34.653, -58.619),
    "Merlo": (-34.665, -58.727),
    "Hurlingham": (-34.595, -58.643),
    "San Miguel": (-34.548, -58.708),
    "Malvinas Argentinas": (-34.505, -58.687),
    "José C. Paz": (-34.516, -58.745),
    "Moreno": (-34.634, -58.791),
    "Ituzaingó": (-34.649, -58.670)
}

# Añadir lat/lon
agg_gba["lat"] = agg_gba["departament_name"].map(lambda x: coords.get(x, (None, None))[0])
agg_gba["lon"] = agg_gba["departament_name"].map(lambda x: coords.get(x, (None, None))[1])
agg_gba = agg_gba.dropna(subset=["lat", "lon"])

# === MAPA FINAL ===

fig = px.choropleth_mapbox(
    merged,
    geojson=merged.geometry.__geo_interface__,
    locations=merged.index,
    color="dengue_favorable",
    hover_name="comuna",
    hover_data={"target_cases": True, "dengue_favorable": True},
    mapbox_style="carto-positron",
    center={"lat": -34.61, "lon": -58.45},
    zoom=9.3,
    opacity=0.5
)

# === BURBUJAS DE CABA ===

# 🔴 Casos reales CABA
fig.add_scattermapbox(
    lat=merged["lat"],
    lon=merged["lon"],
    mode="markers",
    marker=dict(
        size=merged["target_cases"],
        sizemode="area",
        sizeref=2.*merged["target_cases"].max()/(40.**2),
        sizemin=5,
        color="red",
        opacity=0.6
    ),
    name="Casos positivos CABA",
    customdata=merged[["comuna", "target_cases"]],
    hovertemplate=(
        "<b>Comuna %{customdata[0]}</b><br>" +
        "Casos positivos: %{customdata[1]:.0f}<extra></extra>"
    )
)


# 🟠 Semanas favorables CABA
fig.add_scattermapbox(
    lat=merged["lat"],
    lon=merged["lon"],
    mode="markers",
    marker=dict(
        size=merged["dengue_favorable"],
        sizemode="area",
        sizeref=2.*merged["dengue_favorable"].max()/(40.**2),
        sizemin=5,
        color="orange",
        opacity=0.6
    ),
    name="Condiciones favorables CABA",
    customdata=merged[["comuna", "dengue_favorable"]],
    hovertemplate=(
        "<b>Comuna %{customdata[0]}</b><br>" +
        "Semanas favorables: %{customdata[1]:.0f}<extra></extra>"
    )
)

# === BURBUJAS DE GBA ===

# 🔴 Casos reales GBA
fig.add_scattermapbox(
    lat=agg_gba["lat"],
    lon=agg_gba["lon"],
    mode="markers",
    marker=dict(
        size=agg_gba["target_cases"],
        sizemode="area",
        sizeref=3.*agg_gba["target_cases"].max()/(40.**2),
        sizemin=5,
        color="red",
        opacity=0.5
    ),
    name="Casos positivos GBA",
    customdata=agg_gba[["departament_name", "target_cases"]],
    hovertemplate=(
        "<b>%{customdata[0]}</b><br>" +
        "Casos positivos: %{customdata[1]:.0f}<extra></extra>"
    )
)

# 🔵 Semanas favorables GBA
fig.add_scattermapbox(
    lat=agg_gba["lat"],
    lon=agg_gba["lon"],
    mode="markers",
    marker=dict(
        size=agg_gba["dengue_favorable"],
        sizemode="area",
        sizeref=2.*agg_gba["dengue_favorable"].max()/(40.**2),
        sizemin=5,
        color="orange",
        opacity=0.5
    ),
    name="Condiciones favorables GBA",
    customdata=agg_gba[["departament_name", "dengue_favorable"]],
    hovertemplate=(
        "<b>%{customdata[0]}</b><br>" +
        "Semanas favorables: %{customdata[1]:.0f}<extra></extra>"
    )
)

fig.update_layout(
    title="🦟 Mapa Interactivo: Riesgo de Dengue en CABA y GBA",
    margin={"r":0,"t":30,"l":0,"b":0}
)

fig.show()
