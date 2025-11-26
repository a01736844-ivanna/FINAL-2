import streamlit as st
import plotly.io as pio

# ==============Configuración pagina=====================
st.set_page_config(
    page_title="Dashboard Airbnb",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- TEMA GLOBAL DE PLOTLY ----
airbnb_colorway = [
    "#FF5A5F",  # Coral principal
    "#00A699",  # Aqua
    "#484848",  # Gris oscuro
    "#FC642D",  # Naranja
    "#F2C94C",  # Amarillo suave
]

airbnb_sequential = [
    "#F7F3F2",
    "#FAD9D6",
    "#FF7A81",
    "#DE1C40"
]

# Definición del template
custom_theme = dict(
    layout=dict(
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        colorway=airbnb_colorway,
        coloraxis=dict(
            colorscale=airbnb_sequential
        )))

pio.templates["airbnb_pro"] = custom_theme
pio.templates.default = "airbnb_pro"


#====== Definición de páginas=============================
pg_extraccion = st.Page("pages/1_extraccion.py", title="Extracción de Características", icon="🔎")
pg_tablas = st.Page("pages/2_tablas.py", title="Tablas comparativas",icon="📊")
pg_regresiones = st.Page("pages/3_regresiones.py", title="Regresiones", icon="📈")
pg_anova = st.Page("pages/4_ANOVA.py", title= "ANOVA", icon ="📎")


# ============= Agrupar por secciones=====================
nav = st.navigation({
    "Análisis": [pg_extraccion, pg_tablas],
    "Modelado": [pg_regresiones,pg_anova],
})

nav.run()