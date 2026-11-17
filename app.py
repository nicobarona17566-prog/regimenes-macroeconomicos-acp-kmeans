# -*- coding: utf-8 -*-
"""Regímenes macroeconómicos 2010-2023: ACP y K-Means sobre un panel país-año.

Ejecutar con:  streamlit run app.py
"""
import streamlit as st

from regimenes import interfaz as I

st.set_page_config(page_title="¿Ciclo o estructura? · ACP y K-Means", page_icon=":material/stacked_line_chart:",
                   layout="wide", initial_sidebar_state="collapsed")
I.iniciar()
I.estilo()

vistas = [
    st.Page("vistas/pregunta.py", title="Pregunta", icon=":material/help_outline:", default=True),
    st.Page("vistas/datos.py", title="Datos", icon=":material/dataset:"),
    st.Page("vistas/preparacion.py", title="Preparación", icon=":material/tune:"),
    st.Page("vistas/componentes.py", title="Componentes", icon=":material/scatter_plot:"),
    st.Page("vistas/kmeans.py", title="K-Means", icon=":material/workspaces:"),
    st.Page("vistas/tiempo.py", title="Grupos en el tiempo", icon=":material/calendar_month:"),
    st.Page("vistas/trayectorias.py", title="Trayectorias", icon=":material/timeline:"),
    st.Page("vistas/conclusiones.py", title="Conclusiones", icon=":material/fact_check:"),
]
st.navigation(vistas, position="top").run()

st.markdown(
    '<div style="margin-top:3rem;padding-top:.8rem;border-top:1px solid #DDE3E9;color:#7A8692;font-size:.8rem">'
    "Taller individual de ACP y K-Means · Machine Learning aplicado a Economía y Finanzas · Pontificia Universidad Javeriana Cali · "
    "Nicolás Barona · Datos: Banco Mundial, Indicadores del Desarrollo Mundial (CC BY 4.0)</div>",
    unsafe_allow_html=True)
