# pages/1_extraccion.py
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from utils.data_loader import load_all_data

st.title("Extracción de Características")

# ===================== CARGA (externa) =====================
# Esperado: dict {"Barcelona": df, ...}. Si viene lista/tupla, se mapea.
RAW = load_all_data()

def _to_city_map(obj):
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, (list, tuple)):
        # fallback: nombres por defecto según tu proyecto
        default_keys = ["Barcelona", "Cambridge", "Boston", "Hawái", "Budapest"][:len(obj)]
        return dict(zip(default_keys, obj))
    raise ValueError("load_all_data() debe regresar dict o lista/tupla de DataFrames.")

raw_city_map = _to_city_map(RAW)

# ===================== LIMPIEZA =====================
DROP_COLS = [
    "id", "listing_url", "scrape_id", "picture_url", "host_url",
    "host_thumbnail_url", "host_picture_url", "host_about",
    "description", "neighborhood_overview", "amenities",
    "calendar_last_scraped", "first_review", "last_review",
]

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # columnas que no aportan al análisis
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore", inplace=True)
    # eliminar columna fantasma
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    # price a float
    if "price" in df.columns:
        df["price"] = (
            df["price"].astype(str)
            .str.replace(r"[^\d\.\-]", "", regex=True)
            .replace("", np.nan)
            .astype(float)
        )
    return df

dfs_ciudades = {city: clean_df(df) for city, df in raw_city_map.items()}

# ===================== SIDEBAR =====================
show_data = st.sidebar.checkbox("Mostrar Datos")
ciudades_multiselect = st.sidebar.multiselect(
    "Selecciona las Ciudades (Extracción)",
    options=list(dfs_ciudades.keys()),
    default=[k for k in list(dfs_ciudades.keys())[:2]],
    max_selections=4,
)

# ===================== VISTA RÁPIDA =====================
if show_data and ciudades_multiselect:
    st.subheader("Datos de Airbnb (vista rápida)")
    for ciudad in ciudades_multiselect:
        st.markdown(f"**{ciudad}**")
        st.dataframe(dfs_ciudades[ciudad].head(10), use_container_width=True)

# ===================== DEFAULTS =====================
RELEVANT_CATS = [
    "neighbourhood_cleansed", "property_type", "room_type",
    "host_is_superhost", "instant_bookable", "has_availability",
]
NUM_SECTIONS = {
    "💰 Precio y capacidad": ["price", "accommodates", "bedrooms", "beds"],
    "🏡 Disponibilidad y ocupación": ["availability_30", "availability_365", "estimated_occupancy_l365d"],
    "🌟 Evaluaciones y reseñas": ["number_of_reviews", "reviews_per_month", "review_scores_rating"],
}

# ===================== HALLAZGOS =====================
def generar_hallazgos(ciudades):
    lines = []
    for c in ciudades:
        df = dfs_ciudades[c]
        if len(df) < 10:
            continue
        med_price = df["price"].median() if "price" in df.columns else np.nan
        superhosts = df["host_is_superhost"].eq(True).mean() * 100 if "host_is_superhost" in df.columns else np.nan
        rating = df["review_scores_rating"].mean() if "review_scores_rating" in df.columns else np.nan
        occ = df["estimated_occupancy_l365d"].mean() if "estimated_occupancy_l365d" in df.columns else np.nan
        lines.append(
            f"{c} — Mediana precio: {med_price:,.0f} USD, "
            f"Superhosts: {superhosts:.1f}%, Rating promedio: {rating:.1f}, "
            f"Ocupación anual est.: {occ:.1f}%"
        )
    return lines

# ===================== CONTENIDO PRINCIPAL =====================
if not ciudades_multiselect:
    st.warning("Selecciona al menos una ciudad en la barra lateral 👈")
else:
    # ---------- CATEGÓRICAS (DESPLEGABLE) ----------
    with st.expander("🔎 Análisis de variables categóricas", expanded=True):
        ciudad_ref = ciudades_multiselect[0]
        df_ref = dfs_ciudades[ciudad_ref]
        cat_cols = [c for c in df_ref.select_dtypes(include=["object", "category"]).columns if c in RELEVANT_CATS]

        if not cat_cols:
            st.info(f"No hay variables categóricas relevantes en {ciudad_ref}.")
        else:
            cat_var = st.selectbox(
                "Selecciona una variable categórica (aplica a todas las ciudades)",
                options=cat_cols,
                index=0,
            )
            num_categorias = int(df_ref[cat_var].nunique(dropna=True))
            top_n = st.slider(
                "Top categorías a mostrar",
                min_value=1,
                max_value=max(1, num_categorias),
                value=min(10, num_categorias),
                step=1,
            )

            for ciudad in ciudades_multiselect:
                df_ciudad = dfs_ciudades[ciudad]
                if cat_var not in df_ciudad.columns:
                    st.warning(f"**{ciudad}** no tiene la columna **{cat_var}**.")
                    continue

                serie = df_ciudad[cat_var].astype("string").fillna("NA")
                top_vals = serie.value_counts().head(top_n).index
                df_cat = df_ciudad[serie.isin(top_vals)].copy()
                df_cat[cat_var] = df_cat[cat_var].astype("string").fillna("NA")

                st.subheader(f"{ciudad} — {cat_var} (Top-{top_n})")
                col_a, col_b = st.columns(2)

                with col_a:
                    counts = df_cat[cat_var].value_counts().reset_index()
                    counts.columns = [cat_var, "count"]
                    fig1 = px.bar(counts, x="count", y=cat_var, orientation="h",
                                  title=f"Frecuencia de {cat_var} ({ciudad})")
                    fig1.update_layout(yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig1, use_container_width=True)

                with col_b:
                    if "price" in df_ciudad.columns:
                        agg = (
                            df_cat.groupby(cat_var, dropna=False)["price"]
                            .mean().reset_index().rename(columns={"price": "mean_price"})
                        )
                        agg = agg.sort_values("mean_price", ascending=False)
                        fig2 = px.bar(agg, x="mean_price", y=cat_var, orientation="h",
                                      title=f"Precio promedio $USD por {cat_var} ({ciudad})")
                        st.plotly_chart(fig2, use_container_width=True)

    # ---------- NUMÉRICAS (DESPLEGABLE) ----------
    with st.expander("📊 Análisis numérico por ciudad", expanded=False):
        for ciudad in ciudades_multiselect:
            df_ciudad = dfs_ciudades[ciudad]
            st.subheader(ciudad)
            num_cols = df_ciudad.select_dtypes(include="number").columns
            if num_cols.empty:
                st.info("No se detectaron columnas numéricas.")
                continue

            for section_name, section_vars in NUM_SECTIONS.items():
                available_vars = [v for v in section_vars if v in num_cols]
                if not available_vars:
                    continue

                st.markdown(f"#### {section_name}")
                col1, col2 = st.columns(2)

                for i, v in enumerate(available_vars[:4]):
                    target_col = col1 if i % 2 == 0 else col2
                    with target_col:
                        fig_hist = px.histogram(
                            df_ciudad.dropna(subset=[v]),
                            x=v,
                            nbins=30,
                            title=f"{v}"
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)

    # ---------- HALLAZGOS (siempre visibles) ----------
    hallazgos = generar_hallazgos(ciudades_multiselect)
    if hallazgos:
        st.markdown("### 💡 Hallazgos principales")
        st.markdown("""
        <style>
        .hallazgo-box {
            background-color: #f7f7f8;
            border-left: 5px solid #ff5a5f;
            padding: 0.6em 1em;
            margin: 0.3em 0;
            border-radius: 0.4em;
        }
        </style>
        """, unsafe_allow_html=True)
        for h in hallazgos:
            st.markdown(f'<div class="hallazgo-box">{h}</div>', unsafe_allow_html=True)
