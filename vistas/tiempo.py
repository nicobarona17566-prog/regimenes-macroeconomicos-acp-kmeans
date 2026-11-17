# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import streamlit as st

from regimenes import fuente as F
from regimenes import graficos as G
from regimenes import interfaz as I
from regimenes import modelo as M

I.cabeza("Paso 5 · Lectura de los grupos", "Qué distingue a cada grupo y cómo cambia con los años",
         "Los grupos descritos con las variables originales, su peso en cada año, la probabilidad de pasar de uno a otro y su mapa.")
I.linea_estado()

t = I.todo()
d = t["prep"].datos
et = t["km"].etiquetas
nombres, colores, perf = t["nombres"], t["colores"], t["perf"]
vars_ = t["prep"].variables

st.subheader("Perfil de cada grupo en las variables originales")
st.plotly_chart(G.perfil_z(perf, nombres), width="stretch")
I.fuente("Cada celda muestra la mediana del grupo en la unidad original de la variable. El color indica cuán lejos está esa mediana del "
         "promedio del panel, en desviaciones estándar.")
if I.nombrables() and t["k"] in (3, 4):
    med = perf["medianas"]
    g_f = next(g for g, n in nombres.items() if n == M.NOMBRES["fragil"])
    g_p = next(g for g, n in nombres.items() if n == M.NOMBRES["pequeno"])
    g_g = next(g for g, n in nombres.items() if n == M.NOMBRES["grande"])
    pib = perf["pib_pc"]
    I.texto(f"""
<p>{nombres[g_p]}: ingresos y gasto del gobierno cercanos al {I.pct(med.loc[g_p, 'ingreso_publico'], 0)} del PIB, el crecimiento más alto
({I.pct(med.loc[g_p, 'crecimiento'])}) y el desempleo más bajo ({I.pct(med.loc[g_p, 'desempleo'])}). Es un grupo de economías de ingreso
bajo y medio, con un PIB per cápita mediano de US$ {I.ent(pib.loc[g_p])}.</p>
<p>{nombres[g_g]}: el Estado más grande ({I.pct(med.loc[g_g, 'gasto_publico'], 0)} del PIB en gasto), la única mediana positiva de
cuenta corriente ({I.pct(med.loc[g_g, 'cuenta_corriente'])} del PIB) y la inflación más baja. Reúne sobre todo economías de ingreso alto
(PIB per cápita mediano de US$ {I.ent(pib.loc[g_g])}).</p>
<p>{nombres[g_f]}: el desempleo más alto ({I.pct(med.loc[g_f, 'desempleo'])}), el déficit externo más profundo
({I.pct(med.loc[g_f, 'cuenta_corriente'])} del PIB) y un Estado de tamaño intermedio. Es el grupo de Colombia, de Brasil y de buena parte
de América Latina, con un PIB per cápita mediano de US$ {I.ent(pib.loc[g_f])}.</p>
""")

st.subheader("Cada variable, grupo por grupo")
v = st.selectbox("Variable", vars_, format_func=F.etiqueta, index=vars_.index("desempleo") if "desempleo" in vars_ else 0)
st.plotly_chart(G.cajas_grupo(d, et, v, nombres, colores), width="stretch")

st.subheader("El peso de cada grupo, año por año")
rep = M.reparto_por_anio(d, et)
st.plotly_chart(G.reparto_anio(rep, nombres, colores), width="stretch")
n_anio = d.groupby("anio").size()
I.fuente(f"Cada barra suma 100 %. El número de observaciones por año va de {n_anio.min()} ({n_anio.idxmin()}) a {n_anio.max()} ({n_anio.idxmax()}).")
cambio_2020 = (rep.loc[2020] - rep.drop(index=2020).mean()) if 2020 in rep.index else None
if cambio_2020 is not None:
    sube = cambio_2020.idxmax()
    baja = cambio_2020.idxmin()
    I.texto(f"""
<p>Si los grupos fueran puro ciclo, las barras cambiarían de color de un año a otro. Cambian poco. El año que más se aparta es 2020:
«{nombres[sube]}» pasa de un promedio de {I.pct(rep.drop(index=2020)[sube].mean(), 0)} de las observaciones en los demás años a
{I.pct(rep.loc[2020, sube], 0)}, y «{nombres[baja]}» baja de {I.pct(rep.drop(index=2020)[baja].mean(), 0)} a {I.pct(rep.loc[2020, baja], 0)}.
En el resto de la década el reparto se mueve unos pocos puntos de un año a otro.</p>
""")

st.subheader("De un año al siguiente: transiciones")
conteo, prob = M.transiciones(d, et)
st.plotly_chart(G.transiciones(prob, conteo, nombres), width="stretch")
diag = np.diag(prob.to_numpy())
total_pares = int(conteo.to_numpy().sum())
quedan = float(np.trace(conteo.to_numpy()) / total_pares * 100)
I.texto(f"""
<p>La matriz cuenta los pares de años consecutivos de un mismo país ({I.ent(total_pares)} pares) y muestra a qué grupo pasa cada
observación al año siguiente. En el {I.pct(quedan)} de los casos el país sigue en el mismo grupo; la permanencia va de
{I.pct(diag.min(), 0)} en «{nombres[prob.index[int(np.argmin(diag))]]}» a {I.pct(diag.max(), 0)} en «{nombres[prob.index[int(np.argmax(diag))]]}».
Una permanencia tan alta es la firma de la estructura: el grupo de un país en 2015 predice muy bien su grupo en 2016.</p>
""")

st.subheader("Mapa de los grupos en un año")
anios = sorted(d["anio"].unique())
anio = st.select_slider("Año", options=anios, value=2020 if 2020 in anios else anios[-1])
st.plotly_chart(G.mapa(d, et, int(anio), nombres, colores), width="stretch")
I.fuente(f"Países con las variables completas en {anio}: {int((d['anio'] == anio).sum())}. En gris, países sin datos suficientes ese año.")

st.subheader("Grupos según el nivel de ingreso y la región")
c1, c2 = st.columns(2)
ing = pd.crosstab(d["ingreso"], pd.Series([nombres[g] for g in et], name="Grupo"), normalize="index") * 100
ing = ing.reindex([i for i in F.ORDEN_INGRESO if i in ing.index])
c1.markdown("Por nivel de ingreso (% de las observaciones de cada nivel)")
c1.dataframe(I.tabla(ing, 0), width="stretch")
reg = pd.crosstab(d["region"].map(lambda x: F.REGION_CORTA.get(x, x)), pd.Series([nombres[g] for g in et], name="Grupo"), normalize="index") * 100
c2.markdown("Por región (% de las observaciones de cada región)")
c2.dataframe(I.tabla(reg, 0), width="stretch")
I.fuente("El nivel de ingreso y la región no entraron al modelo; que los grupos se ordenen según ellos es un resultado.")
