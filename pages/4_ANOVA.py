# pages/4_anova_manova.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA

from utils.data_loader import load_all_data


st.title("ANOVA y MANOVA")

# ===================== CARGA Y LIMPIEZA =====================
RAW = load_all_data()

def _to_city_map(obj):
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, (list, tuple)):
        default_keys = ["Barcelona", "Cambridge", "Boston", "Hawái", "Budapest"][:len(obj)]
        return dict(zip(default_keys, obj))
    raise ValueError("load_all_data() debe regresar dict o lista/tupla de DataFrames.")


def clean_city_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Remover columnas inútiles
    drop_cols = [
        "host_id","scrape_id","listing_url","host_url","picture_url","host_thumbnail_url",
        "host_picture_url","last_scraped","calendar_last_scraped","source","calendar_updated",
        "description","name","license","host_about","neighborhood_overview","bathrooms_text",
        "amenities","host_verifications","first_review","last_review"
    ]
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Quitar columnas "Unnamed"
    for c in list(df.columns):
        if c.lower().startswith("unnamed"):
            df = df.drop(columns=[c])

    # Limpieza price
    if "price" in df.columns:
        df["price"] = (
            df["price"].astype(str)
            .str.replace(r"[^\d\.\-]", "", regex=True)
            .replace("", np.nan)
            .astype(float)
        )
    return df


raw_city_map = _to_city_map(RAW)
dfs_ciudades = {k: clean_city_df(v) for k, v in raw_city_map.items()}

# ===================== SIDEBAR =====================
st.sidebar.header("ANOVA / MANOVA")

ciudades_sel = st.sidebar.multiselect(
    "Selecciona hasta 3 ciudades",
    options=list(dfs_ciudades.keys()),
    default=[list(dfs_ciudades.keys())[0]],
    max_selections=3
)

if len(ciudades_sel) == 0:
    st.warning("Selecciona al menos 1 ciudad para continuar.")
    st.stop()

st.subheader(f"Ciudades seleccionadas: {', '.join(ciudades_sel)}")


df_list = []
for ciudad in ciudades_sel:
    df_tmp = dfs_ciudades[ciudad].copy()
    df_tmp["__city__"] = ciudad  
    df_list.append(df_tmp)

df = pd.concat(df_list, ignore_index=True)


# ===================== VARIABLES RELEVANTES =====================
categoricas_utiles = [ "id",
    "host_is_superhost",
    "host_identity_verified",
    "host_has_profile_pic",
    "instant_bookable",
    "neighbourhood_cleansed",
    "neighbourhood_group_cleansed",
    "property_type",
    "room_type",
    "has_availability"
]

numericas_utiles = [
    "price",
    "accommodates",
    "bedrooms",
    "bathrooms",
    "beds",
    "minimum_nights",
    "maximum_nights",
    "availability_30",
    "availability_90",
    "availability_365",
    "number_of_reviews",
    "number_of_reviews_ltm",
    "estimated_occupancy_l365d",
    "estimated_revenue_l365d",
    "review_scores_rating",
    "reviews_per_month",
]

categoricas = [c for c in categoricas_utiles if c in df.columns]
numericas = [n for n in numericas_utiles if n in df.columns]

if not numericas or not categoricas:
    st.error("No hay suficientes variables útiles para ANOVA/MANOVA en esta ciudad.")
    st.stop()

# ===================== CONTROLES ANOVA =====================
st.sidebar.markdown("### ANOVA (un factor)")

var_dep_anova = st.sidebar.selectbox(
    "Variable dependiente (numérica)",
    options=numericas,
    index=numericas.index("price") if "price" in numericas else 0,
)

var_cat_anova = st.sidebar.selectbox(
    "Factor categórico",
    options=categoricas
)

# ===================== CONTROLES MANOVA =====================
st.sidebar.markdown("### MANOVA (multivariado)")

vars_dep_manova = st.sidebar.multiselect(
    "Variables dependientes (numéricas)",
    options=numericas,
    default=["price"] if "price" in numericas else numericas[:1],
)

vars_ind_manova = st.sidebar.multiselect(
    "Factores categóricos",
    options=categoricas
)

# ===================== ANOVA =====================
st.markdown("## ANOVA — un factor")

df_anova = df[[var_dep_anova, var_cat_anova]].dropna()

if df_anova.empty or df_anova[var_cat_anova].nunique() < 2:
    st.warning("No se puede ejecutar ANOVA con estas variables.")
else:
    formula = f"{var_dep_anova} ~ C({var_cat_anova})"
    modelo = ols(formula, data=df_anova).fit()
    tabla = sm.stats.anova_lm(modelo, typ=2)

    st.write("### Tabla ANOVA")
    st.dataframe(tabla)

    fig = px.box(
        df_anova, x=var_cat_anova, y=var_dep_anova,
        color=var_cat_anova, title=f"{var_dep_anova} por {var_cat_anova}"
    )
    st.plotly_chart(fig, use_container_width=True)

# ===================== MANOVA =====================
st.markdown("---")
st.markdown("## MANOVA — multivariado")

if vars_dep_manova and vars_ind_manova:
    df_m = df[vars_dep_manova + vars_ind_manova].dropna()

    if df_m.empty:
        st.warning("No hay datos suficientes.")
    else:
        dep = " + ".join(vars_dep_manova)
        ind = " + ".join([f"C({v})" for v in vars_ind_manova])
        formula_m = f"{dep} ~ {ind}"

        try:
            manova = MANOVA.from_formula(formula_m, data=df_m)
            st.write("### Tabla MANOVA")
            st.write(manova.mv_test())

            
            if len(vars_dep_manova) >= 2:
                fig2 = px.scatter(
                    df_m,
                    x=vars_dep_manova[0],
                    y=vars_dep_manova[1],
                    color=vars_ind_manova[0],
                    title=f"{vars_dep_manova[0]} vs {vars_dep_manova[1]}"
                )
                st.plotly_chart(fig2, use_container_width=True)

        except Exception as e:
            st.error(f"Error ejecutando MANOVA: {e}")
else:
    st.info("Selecciona variables en el sidebar para ejecutar MANOVA.")
