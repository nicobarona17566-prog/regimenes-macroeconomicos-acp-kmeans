# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import streamlit as st

from regimenes import fuente as F
from regimenes import graficos as G
from regimenes import interfaz as I
from regimenes import modelo as M

I.cabeza("Paso 6 · Trayectorias", "El recorrido de cada país",
         "Elige un país para ver en qué grupo cayó cada año, cómo se movió en el plano de los componentes y a quién se parece.")

t0 = I.todo()
d0 = t0["prep"].datos
opciones = d0.drop_duplicates("iso2").sort_values("pais")
iso_a_pais = dict(zip(opciones["iso2"], opciones["pais"]))
if I.estado()["pais"] not in iso_a_pais:
    I.estado()["pais"] = "co" if "co" in iso_a_pais else opciones["iso2"].iloc[0]
    st.session_state.pop("w_pais", None)
I.estado()["comparar"] = [i for i in I.estado()["comparar"] if i in iso_a_pais]
if any(i not in iso_a_pais for i in st.session_state.get("w_comparar", [])):
    st.session_state.pop("w_comparar", None)

c1, c2 = st.columns([1, 2])
with c1:
    I.control(st.selectbox, "pais", "País", options=list(iso_a_pais), format_func=iso_a_pais.get)
with c2:
    I.control(st.multiselect, "comparar", "Comparar con", options=list(iso_a_pais), format_func=iso_a_pais.get, max_selections=5)
I.linea_estado()

t = I.todo()
d, et, S = t["prep"].datos, t["km"].etiquetas, t["S"]
nombres, colores = t["nombres"], t["colores"]
foco = I.estado()["pais"]
comp = [i for i in I.estado()["comparar"] if i != foco]
isos = [foco] + comp
nom_foco = iso_a_pais[foco]

tr = M.trayectoria(d, et, S, foco)
n_cambios = M.cambios(tr)
moda = tr["grupo"].mode().iloc[0]
I.datos_clave([
    (str(len(tr)), f"años de {nom_foco} con las variables completas"),
    (str(n_cambios), "cambios de grupo entre años consecutivos"),
    (I.pct((tr["grupo"] == moda).mean() * 100, 0), f"de sus años en «{nombres[moda]}»"),
    (I.num(float(t["km"].silueta_obs[(d['iso2'] == foco).to_numpy()].mean()), 3), "silueta media de sus observaciones"),
])
st.write("")

st.subheader("Grupo de cada año")
anios = list(range(int(d["anio"].min()), int(d["anio"].max()) + 1))
st.plotly_chart(G.cinta(d, et, isos, nombres, colores, anios), width="stretch")
st.markdown(I.leyenda_grupos(t), unsafe_allow_html=True)
I.fuente("Las celdas vacías son años en que al país le falta alguna de las variables del modelo.")

if foco == "co" and I.es_estudio():
    cl = M.trayectoria(d, et, S, "cl")
    paso_cl = cl.loc[cl["grupo"] == moda, "anio"].min() if (cl["grupo"] == moda).any() else None
    I.texto(f"""
<p>Colombia pasa los {len(tr)} años en el mismo grupo, «{nombres[moda]}», incluso en 2020. La pandemia la movió dentro del plano, como se ve
abajo, pero no lo bastante para cruzar la frontera: su desempleo y su déficit externo ya eran los de ese grupo antes del choque.
Chile hace el recorrido contrario al de la quietud colombiana: pasa del grupo de Estado pequeño al de Colombia desde {paso_cl}, cuando su
desempleo sube y su cuenta corriente se deteriora.</p>
""")

st.subheader("Recorrido en el plano de los componentes")
r = t["acp"]
st.plotly_chart(G.trayectorias(d, S if S.shape[1] >= 2 else r.scores, isos, iso_a_pais, (r.varianza[0] * 100, r.varianza[1] * 100)),
                width="stretch")
I.fuente(f"Fondo gris: todas las observaciones del panel. Cada línea une los años de un país; se rotulan el primero, el último y 2020. "
         f"El plano resume el {I.pct(r.acumulada[1] * 100)} de la varianza.")

st.subheader(f"Los datos de {nom_foco}, año por año")
vista = tr[["anio"] + t["prep"].variables].copy()
vista.insert(1, "Grupo", [nombres[g] for g in tr["grupo"]])
vista = vista.rename(columns={"anio": "Año", **{v: F.etiqueta(v) for v in t["prep"].variables}})
st.dataframe(I.tabla(vista, 1), hide_index=True, width="stretch")

st.subheader("A quién se parece")
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"En todo el periodo (punto promedio de {nom_foco} en los componentes)")
    gp = M.gemelos_periodo(d, S, foco)
    if len(gp):
        st.dataframe(I.tabla(gp[["pais", "region", "anios", "distancia"]].rename(columns={"pais": "País", "region": "Región", "anios": "Años",
                     "distancia": "Distancia"}), 2), hide_index=True, width="stretch")
with c2:
    anios_foco = tr["anio"].tolist()
    anio = st.select_slider("En un año", options=anios_foco, value=anios_foco[-1])
    ga = M.gemelos_anio(d, S, foco, int(anio))
    if len(ga):
        st.dataframe(I.tabla(ga[["pais", "region", "distancia"]].rename(columns={"pais": "País", "region": "Región", "distancia": "Distancia"}), 2),
                     hide_index=True, width="stretch")
I.fuente("Distancia euclidiana en el espacio de los componentes conservados. Para el periodo se exigen al menos cinco años con datos.")

ac = M.acompanantes(d, et, foco)
if len(ac):
    siempre = ac[ac >= 99.9].index.tolist()
    I.texto(f"""
<p>Otra forma de medir el parecido es contar en cuántos de los años de {nom_foco} otro país cayó en su mismo grupo. Comparten grupo con
{nom_foco} en todos los años en que ambos tienen datos: {", ".join(siempre[:12]) if siempre else "ningún país"}{"…" if len(siempre) > 12 else ""}.</p>
""")
    with st.expander("Ver el porcentaje de años compartidos con cada país"):
        st.dataframe(I.tabla(ac.rename("% de años en el mismo grupo").to_frame(), 0), width="stretch")
