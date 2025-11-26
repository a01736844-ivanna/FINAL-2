# pages/3_regresiones.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score,
    accuracy_score, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
from utils.data_loader import load_all_data

st.title("Regresión")

# ===================== CARGA Y LIMPIEZA =====================
RAW = load_all_data()

def _to_city_map(obj):
    """Convierte la salida de load_all_data en {ciudad: DataFrame}."""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, (list, tuple)):
        default_keys = ["Barcelona", "Cambridge", "Boston", "Hawái", "Budapest"][:len(obj)]
        return dict(zip(default_keys, obj))
    raise ValueError("load_all_data() debe regresar dict o lista/tupla de DataFrames.")

def clean_city_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # columna fantasma
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    # normalizar price a numérico
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
ciudades_reg_sel = st.sidebar.multiselect(
    "Ciudades para regresión",
    options=list(dfs_ciudades.keys()),
    default=[k for k in list(dfs_ciudades.keys())[:2]],
    max_selections=4
)

tipo_reg = st.sidebar.radio(
    "Tipo de regresión",
    options=[
        "Regresión lineal simple",
        "Regresión lineal múltiple",
        "Regresión no lineal",
        "Regresión logística"   # ← NUEVO
    ],
    index=0
)

if not ciudades_reg_sel:
    st.warning("Selecciona al menos una ciudad en la barra lateral 👈")
    st.stop()

# ===================== REGRESIÓN LINEAL (simple / múltiple) =====================
if tipo_reg in ("Regresión lineal simple", "Regresión lineal múltiple"):
    n = len(ciudades_reg_sel)
    cols = st.columns(min(3, n))

    for i, ciudad in enumerate(ciudades_reg_sel):
        if i > 0 and i % 3 == 0:
            cols = st.columns(min(3, n - i))

        df_ciudad = dfs_ciudades[ciudad]
        with cols[i % 3]:
            st.subheader(ciudad)

            if "price" not in df_ciudad.columns:
                st.info("Falta columna 'price'.")
                continue

            # ---------- Lineal simple: price ~ accommodates ----------
            if tipo_reg == "Regresión lineal simple":
                if "accommodates" not in df_ciudad.columns:
                    st.info("Falta columna 'accommodates'.")
                    continue

                tmp = df_ciudad[["accommodates", "price"]].astype(float).dropna()
                if len(tmp) < 3:
                    st.info("Datos insuficientes para ajustar el modelo.")
                    continue

                x = tmp["accommodates"].to_numpy()
                y = tmp["price"].to_numpy()
                a, b = np.polyfit(x, y, 1)  # β1, β0
                y_pred = a * x + b
                r2 = r2_score(y, y_pred)

                m1, m2, m3 = st.columns(3)
                m1.metric("R²", f"{r2:.3f}")
                m2.metric("Pendiente (β1)", f"{a:.3f}")
                m3.metric("Intercepto (β0)", f"{b:.2f}")

                fig = px.scatter(
                    tmp, x="accommodates", y="price",
                    labels={"accommodates": "Accommodates", "price": "Price"},
                    title="Price ~ Accommodates (Regresión lineal)"
                )
                x_line = np.linspace(x.min(), x.max(), 50)
                fig.add_trace(go.Scatter(x=x_line, y=a * x_line + b, mode="lines", name="Ajuste"))
                fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)

            # ---------- Lineal múltiple ----------
            elif tipo_reg == "Regresión lineal múltiple":
                num_cols = df_ciudad.select_dtypes(include="number").columns.tolist()
                num_cols = [c for c in num_cols if not c.lower().startswith("unnamed")]
                candidatas = [c for c in num_cols if c != "price"]

                if len(candidatas) < 2:
                    st.info("Se requieren al menos 2 variables numéricas distintas a 'price' en el dataset.")
                    continue

                default_preds = []
                if "accommodates" in candidatas:
                    default_preds.append("accommodates")
                for c in candidatas:
                    if c not in default_preds:
                        default_preds.append(c)
                    if len(default_preds) >= 2:
                        break

                predictores = st.multiselect(
                    "Variables explicativas (elige 2 o más)",
                    options=candidatas,
                    default=default_preds,
                    key=f"preds_mult_{ciudad}"
                )

                if len(predictores) < 2:
                    st.warning("Selecciona al menos 2 variables para ajustar la regresión múltiple.")
                    continue

                tmp = df_ciudad[predictores + ["price"]].dropna()
                if len(tmp) < len(predictores) + 1:
                    st.info("Datos insuficientes después de eliminar NA para ajustar el modelo.")
                    continue

                X = tmp[predictores].to_numpy(dtype=float)
                y = tmp["price"].to_numpy(dtype=float)

                modelo = LinearRegression()
                modelo.fit(X, y)

                r2 = modelo.score(X, y)
                intercepto = modelo.intercept_
                coefs = modelo.coef_

                m1, m2, m3 = st.columns(3)
                m1.metric("R² (múltiple)", f"{r2:.3f}")
                m2.metric("N° predictores", f"{len(predictores)}")
                m3.metric("Intercepto (β0)", f"{intercepto:.2f}")

                coef_df = pd.DataFrame({
                    "Variable": predictores,
                    "Coeficiente (β)": coefs
                })
                st.dataframe(coef_df, use_container_width=True)

                # Efecto parcial de una variable manteniendo las demás en su media
                var_plot = "accommodates" if "accommodates" in predictores else predictores[0]

                fig = px.scatter(
                    tmp[[var_plot, "price"]], x=var_plot, y="price",
                    labels={var_plot: var_plot, "price": "Price"},
                    title=f"Price ~ {var_plot} (Regresión múltiple, otros = media)"
                )

                x_line = np.linspace(tmp[var_plot].min(), tmp[var_plot].max(), 50)
                medias = tmp[predictores].mean()
                X_line = np.tile(medias.to_numpy(), (50, 1))
                idx_plot = predictores.index(var_plot)
                X_line[:, idx_plot] = x_line
                y_line = modelo.predict(X_line)

                fig.add_trace(go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name="Predicción (otros = media)"
                ))
                st.plotly_chart(fig, use_container_width=True)


# ===================== REGRESIÓN NO LINEAL=====================
if tipo_reg == "Regresión no lineal":
    st.markdown("## Regresión no lineal")

    # ---- Definición de funciones ----
    def f_quadratic(x, a, b, c):          # a*x^2 + b*x + c
        return a*x**2 + b*x + c

    def f_exp_decay(x, a, b, c):          # a*exp(-b*x) + c
        return a*np.exp(-b*x) + c

    def f_abs_linear(x, a, b, c):         # a*|x| + b*x + c
        return a*np.abs(x) + b*x + c

    def f_rational_poly2(x, a, b, c):     # (a*x^2 + b)/(c*x^2)
        denom = c*(x**2)
        return (a*x**2 + b) / np.where(denom == 0, 1e-9, denom)

    def f_log(x, a, b):                   # a*log(x) + b
        return a*np.log(x) + b

    def f_inv_quad(x, a):                 # (1/a)*x^2
        return (1.0 / np.where(a == 0, 1e-9, a)) * (x**2)

    def f_inv_poly2(x, a, b, c):          # (a/b)*x^2 + c*x
        return (a / np.where(b == 0, 1e-9, b)) * (x**2) + c*x

    MODELS = {
        "Función cuadrática (a*x^2 + b*x + c)":           (f_quadratic,      [1.0, 1.0, 0.0]),
        "Función exponencial (a*exp(-b*x)+c)":            (f_exp_decay,      [1.0, 0.1, 0.0]),
        "Valor absoluto (a*|x| + b*x + c)":               (f_abs_linear,     [1.0, 0.0, 0.0]),
        "Cociente polinomios ((a*x^2+b)/(c*x^2))":        (f_rational_poly2, [1.0, 1.0, 1.0]),
        "Logarítmica (a*log(x)+b)":                       (f_log,            [1.0, 0.0]),
        "Cuadrática inversa ((1/a)*x^2)":                 (f_inv_quad,       [1.0]),
        "Polinomial inversa ((a/b)*x^2 + c*x)":           (f_inv_poly2,      [1.0, 1.0, 0.0]),
    }

    # ---- SIDEBAR ----
    df_ref_nl = dfs_ciudades[ciudades_reg_sel[0]]

    # variables numéricas relevantes para el modelo (no ids, no scrapes)
    candidatos_xy = [
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

    numeric_cols = [c for c in candidatos_xy if c in df_ref_nl.columns]

    if not numeric_cols:
        st.warning("No hay suficientes variables numéricas adecuadas para la regresión no lineal.")
        st.stop()

    # Y puede ser price o alguna de las numéricas elegidas
    y_candidates = []
    if "price" in df_ref_nl.columns:
        y_candidates.append("price")
    y_candidates += numeric_cols

    x_col = st.sidebar.selectbox(
        "Variable X (predictor)",
        options=numeric_cols,
        index=numeric_cols.index("accommodates") if "accommodates" in numeric_cols else 0,
        key="nl_x"
    )

    y_col = st.sidebar.selectbox(
        "Variable Y (objetivo)",
        options=y_candidates,
        index=y_candidates.index("price") if "price" in y_candidates else 0,
        key="nl_y"
    )

    model_name = st.sidebar.selectbox(
        "Modelo no lineal",
        options=list(MODELS.keys()),
        key="nl_model"
    )
    func, p0_default = MODELS[model_name]

    # ---- Visual por ciudad ----
    n = len(ciudades_reg_sel)
    cols_nl = st.columns(min(4, n))

    for i, ciudad in enumerate(ciudades_reg_sel):
        if i > 0 and i % 4 == 0:
            cols_nl = st.columns(min(4, n - i))

        with cols_nl[i % 4]:
            st.subheader(ciudad)

            dfc = dfs_ciudades[ciudad].copy()
            if x_col not in dfc.columns or y_col not in dfc.columns:
                st.info("Columnas seleccionadas no disponibles.")
                continue

            dfc = dfc[[x_col, y_col]].dropna()
            if dfc.empty:
                st.info("Sin datos suficientes.")
                continue

            x_raw = dfc[x_col].astype(float).to_numpy()
            y = dfc[y_col].astype(float).to_numpy()

            # Filtros de dominio básicos
            mask = np.isfinite(x_raw) & np.isfinite(y)
            if "Logarítmica" in model_name:
                mask &= x_raw > 0
            if "Cociente polinomios" in model_name:
                mask &= x_raw != 0

            x_raw, y = x_raw[mask], y[mask]
            if len(x_raw) < 5:
                st.info("Datos insuficientes tras filtros de dominio.")
                continue

            x = x_raw.copy()

            # Ajuste de la curva
            try:
                popt, _ = curve_fit(func, x, y, p0=p0_default, maxfev=12000)
                y_hat = func(x, *popt)
                r2 = 1 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2)
            except Exception as e:
                st.warning(f"No se pudo ajustar el modelo: {e}")
                continue

            st.metric("R²", f"{r2:.3f}")

            # Curva para graficar
            order = np.argsort(x_raw)
            x_plot = x_raw[order]
            y_plot = func(x_plot, *popt)

            fig = px.scatter(
                x=x_raw, y=y,
                labels={"x": x_col, "y": y_col},
                title=f"{ciudad} — {model_name}"
            )
            fig.add_trace(go.Scatter(x=x_plot, y=y_plot, mode="lines", name="Ajuste"))
            fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)





# ===================== REGRESIÓN LOGÍSTICA =====================
if tipo_reg == "Regresión logística":
    st.markdown("## Regresión logística")

    # Tomamos una ciudad de referencia solo para saber qué columnas existen
    ciudad_ref = ciudades_reg_sel[0]
    df_ref = dfs_ciudades[ciudad_ref]

    # Asegurar que exista price
    if "price" not in df_ref.columns:
        st.warning("No se puede hacer regresión logística: falta columna 'price'.")
    else:

        # --------- Variables numéricas + host_is_superhost ----------
        posibles_vars = [
            "accommodates",
            "bedrooms",
            "bathrooms",
            "minimum_nights",
            "availability_365",
            "number_of_reviews",
            "estimated_occupancy_l365d",
            "estimated_revenue_l365d",
            "review_scores_rating",
            "reviews_per_month",
            "host_is_superhost"    # 🔥 AGREGADA AQUÍ
        ]

        # Filtrar solo las que existan
        vars_disponibles = [v for v in posibles_vars if v in df_ref.columns]

        # Selector para predictores (como pclass, age, fare en Titanic)
        predictores_log = st.sidebar.multiselect(
            "Variables explicativas (logística)",
            options=vars_disponibles,
            default=vars_disponibles[:3],
            key="logit_vars_airbnb",
        )

        if len(predictores_log) == 0:
            st.info("Selecciona al menos una variable explicativa para la regresión logística.")
        else:

            # Hasta 4 ciudades por fila
            n = len(ciudades_reg_sel)
            cols_log = st.columns(min(4, n))

            for i, ciudad in enumerate(ciudades_reg_sel):
                if i > 0 and i % 4 == 0:
                    cols_log = st.columns(min(4, n - i))

                with cols_log[i % 4]:
                    st.subheader(ciudad)

                    df_city = dfs_ciudades[ciudad].copy()

                    if "price" not in df_city.columns:
                        st.info("Falta 'price' en esta ciudad.")
                        continue

                    # ---------------- Conversión host_is_superhost → binaria ----------------
                    if "host_is_superhost" in df_city.columns:
                        mapa_super = {
                            "t": 1, "f": 0,
                            "T": 1, "F": 0,
                            "True": 1, "False": 0,
                            True: 1, False: 0
                        }
                        df_city["host_is_superhost_bin"] = (
                            df_city["host_is_superhost"].map(mapa_super).fillna(0).astype(int)
                        )

                        # Si el usuario eligió host_is_superhost, reemplazamos por la variable binaria:
                        predictores_uso = []
                        for p in predictores_log:
                            if p == "host_is_superhost":
                                predictores_uso.append("host_is_superhost_bin")
                            else:
                                predictores_uso.append(p)
                    else:
                        predictores_uso = predictores_log

                    # ---------------- Variable objetivo (high_price) ----------------
                    umbral_local = df_city["price"].median()
                    df_city["high_price"] = (df_city["price"] > umbral_local).astype(int)

                    # Limpiar dataset para X y y
                    cols_necesarias = predictores_uso + ["high_price"]
                    df_city = df_city[cols_necesarias].dropna()

                    if df_city.empty:
                        st.info("Datos insuficientes después de limpiar NaN.")
                        continue

                    X = df_city[predictores_uso].to_numpy(dtype=float)
                    y = df_city["high_price"].to_numpy(dtype=int)

                    if y.sum() == 0 or y.sum() == len(y):
                        st.info("La variable objetivo tiene solo una clase en esta ciudad.")
                        continue

                    # ---------------- Train-test split ----------------
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.3, random_state=None
                    )

                    # ---------------- Escalamiento (StandardScaler) ----------------
                    escalar = StandardScaler()
                    X_train = escalar.fit_transform(X_train)
                    X_test = escalar.transform(X_test)

                    # ---------------- Modelo Logístico ----------------
                    modelo_log = LogisticRegression(max_iter=1000)
                    modelo_log.fit(X_train, y_train)
                    y_pred = modelo_log.predict(X_test)

                    # ---------------- Métricas ----------------
                   
                    cm = confusion_matrix(y_test, y_pred)
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, zero_division=0)
                    rec = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Accuracy", f"{acc:.3f}")
                    m2.metric("Precision (1)", f"{prec:.3f}")
                    m3.metric("Recall (1)", f"{rec:.3f}")
                    m4.metric("F1 Score", f"{f1:.3f}")

                    # ---------------- Matriz de confusión ----------------
                    cm_df = pd.DataFrame(
                        cm,
                        index=["Real 0 (barato)", "Real 1 (caro)"],
                        columns=["Pred 0", "Pred 1"]
                    )

                    fig_cm = px.imshow(
                        cm_df,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale=["#DEDEDE", "#FFB3B3", "#FF5A5F", "#00999F"],
                    )
                    fig_cm.update_layout(
                        title="Matriz de confusión",
                        margin=dict(l=10, r=10, t=40, b=10),
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
