import streamlit as st
import pandas as pd

@st.cache_resource
def load_all_data():
    dfb = pd.read_csv("/Users/hijos/Desktop/Analitica Prof/Prof. Alfredo/FINAL 2/data/Barcelona_Limpios.csv")
    dfc = pd.read_csv("/Users/hijos/Desktop/Analitica Prof/Prof. Alfredo/FINAL 2/data/Cambridge_Limpios.csv")
    dfbo = pd.read_csv("/Users/hijos/Desktop/Analitica Prof/Prof. Alfredo/FINAL 2/data/Boston_Limpios.csv")
    dfh = pd.read_csv("/Users/hijos/Desktop/Analitica Prof/Prof. Alfredo/FINAL 2/data/Hawai_Limpios.csv")
    dfbu = pd.read_csv("/Users/hijos/Desktop/Analitica Prof/Prof. Alfredo/FINAL 2/data/Budapest_Limpios.csv")
    return dfb, dfc, dfbo, dfh, dfbu