# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import streamlit as st

from regimenes import fuente as F
from regimenes import graficos as G
from regimenes import interfaz as I
from regimenes import modelo as M

I.cabeza("Paso 4 · Segmentación con K-Means", "¿Cuántos grupos hay?",
         "Codo, silueta y estabilidad para elegir K; los grupos en el plano y la calidad de la asignación de cada observación.")

with st.expander("Cómo funciona K-Means y cómo se evalúa"):
    I.texto("""
<p>K-Means reparte las observaciones en K grupos de modo que la suma de distancias al cuadrado entre cada punto y el centro de su grupo
(la inercia) sea mínima. Empieza con K centros, asigna cada punto al más cercano, recalcula los centros como promedios y repite hasta
que nada cambia. Como el resultado depende del punto de partida, se corre diez veces con centros distintos y se queda la mejor
solución. Aquí trabaja sobre las coordenadas de los componentes conservados, que ya están en una escala común.</p>
<p>Se usan tres criterios. El codo busca el K a partir del cual la inercia deja de caer mucho. La silueta compara, para cada observación,
la distancia promedio a su grupo (a) con la del grupo vecino más cercano (b): s = (b - a) / max(a, b); va de -1 a 1, y Kaufman y
Rousseeuw (1990) leen 0,26-0,50 como estructura débil y 0,51-0,70 como razonable. La estabilidad repite K-Means sobre 20 submuestras del
80 % y compara cada partición con la del panel completo mediante el índice de Rand ajustado (ARI): 1 es la misma partición y 0 es lo que
daría el azar. Un K con buena silueta pero baja estabilidad describe grupos que dependen de qué observaciones entren.</p>
""")

t0 = I.todo()
tk = I.tabla_k(t0["a"], t0["nc"])

st.subheader("Comparación de valores de K")
I.control(st.slider, "k", "Número de grupos (K)", min_value=2, max_value=8, step=1)
t = I.todo()
k = t["k"]
I.linea_estado()
st.plotly_chart(G.curvas_k(tk, k), width="stretch")
vista = tk.rename(columns={"k": "K", "inercia": "Inercia", "silueta": "Silueta", "estabilidad": "Estabilidad (ARI medio)",
                           "estabilidad_min": "Estabilidad mínima", "caida_inercia": "Caída de la inercia (%)"})
st.dataframe(I.tabla(vista, 3, {"K": 0, "Inercia": 1, "Caída de la inercia (%)": 1}), hide_index=True, width="stretch")

if I.prep_es_estudio() and t["nc"] == I.N_COMP_ESTUDIO:
    f = tk.set_index("k")
    I.texto(f"""
<p>La silueta más alta es la de K = 2 ({I.num(f.loc[2, 'silueta'], 3)}), pero dos grupos solo separan Estados grandes de Estados
pequeños y no dicen nada de la situación macroeconómica. Desde K = 3 la silueta cae poco a poco, sin un salto que marque un
óptimo, y la inercia tampoco forma un codo nítido: pasar de 2 a 3 grupos la reduce un {I.pct(f.loc[3, 'caida_inercia'], 0)} y de 3 a 4 un
{I.pct(f.loc[4, 'caida_inercia'], 0)}.</p>
<p>La estabilidad desempata. Con K = 3, las particiones hechas sobre submuestras coinciden con la del panel completo con un ARI medio de
{I.num(f.loc[3, 'estabilidad'], 2)}, y la silueta ({I.num(f.loc[3, 'silueta'], 3)}) es la más alta desde K = 3. Con K = 4 aparece un grupo nuevo,
la recesión de 2020, pero la estabilidad baja a {I.num(f.loc[4, 'estabilidad'], 2)} y en la peor submuestra a
{I.num(f.loc[4, 'estabilidad_min'], 2)}: ese cuarto grupo existe en unas muestras y en otras no. El estudio elige K = 3 y deja la partición de
cuatro grupos como contraste, que se puede ver moviendo el deslizador.</p>
<p>Con cualquier K, la silueta queda en la franja de estructura débil de Kaufman y Rousseeuw. Las observaciones forman una nube continua
con zonas más densas, no islas separadas, y los grupos deben leerse como perfiles típicos con fronteras difusas.</p>
""")

st.subheader("Los grupos")
perf = t["perf"]
nombres = t["nombres"]
colores = t["colores"]
d = t["prep"].datos
et = t["km"].etiquetas
cols = st.columns(min(k, 4))
for g in sorted(nombres):
    z = perf["z"].loc[g]
    med = perf["medianas"].loc[g]
    tipicos = d.assign(g=et).loc[lambda x: x["g"] == g, "pais"].value_counts().head(4).index.tolist()
    pib = perf["pib_pc"].loc[g] if perf["pib_pc"] is not None else np.nan
    with cols[g % len(cols)]:
        st.markdown(
            f'<div class="regimen"><h4><span class="sw" style="background:{colores[g]}"></span>{nombres[g]}</h4>'
            f'<div class="n">{I.ent(perf["tam"].loc[g])} observaciones · {perf["paises"].loc[g]} países · PIB per cápita mediano US$ {I.ent(pib)}</div>'
            f'<p>Rasgos: {M.rasgos(z)}. Medianas: inflación {I.pct(med.get("inflacion", np.nan))}, crecimiento {I.pct(med.get("crecimiento", np.nan))}, '
            f'desempleo {I.pct(med.get("desempleo", np.nan))}, cuenta corriente {I.pct(med.get("cuenta_corriente", np.nan))} del PIB, '
            f'ingreso público {I.pct(med.get("ingreso_publico", np.nan), 0)} del PIB.</p>'
            f'<p class="n">Países con más años en el grupo: {", ".join(tipicos)}.</p></div>', unsafe_allow_html=True)
        st.write("")
if not I.nombrables():
    I.aviso("Con esta configuración los grupos se identifican con letras. Las letras no implican orden: A es solo el grupo más grande.")

st.subheader("Los grupos en el plano de los componentes")
r = t["acp"]
etiquetas_txt = pd.Series([nombres[g] for g in et])
fig = G.plano(d, t["S"] if t["nc"] >= 2 else r.scores, etiquetas_txt, {nombres[g]: colores[g] for g in sorted(nombres)},
              {nombres[g]: t["simbolos"][g] for g in nombres}, None, I.estado()["pais"], I.pais(I.estado()["pais"]),
              (r.varianza[0] * 100, r.varianza[1] * 100))
st.plotly_chart(fig, width="stretch")
I.fuente(f"K-Means trabaja con {t['nc']} componentes; el plano muestra los dos primeros ({I.pct(r.acumulada[1] * 100)} de la varianza), así que "
         "algunos puntos que parecen mezclados se separan en el tercer eje.")

st.subheader("Calidad de la asignación: silueta de cada observación")
sil = t["km"].silueta_obs
st.plotly_chart(G.silueta_obs(sil, et, nombres, colores, t["km"].silueta), width="stretch")
neg = d.assign(s=sil, g=[nombres[g] for g in et])
negativas = neg[neg["s"] < 0]
por_g = neg.groupby("g")["s"].agg(["mean", lambda s: (s < 0).mean() * 100])
por_g.columns = ["Silueta media", "% con silueta negativa"]
por_g.index.name = "Grupo"
st.dataframe(I.tabla(por_g, 3, {"% con silueta negativa": 1}), width="stretch")
I.texto(f"""
<p>{I.ent(len(negativas))} observaciones ({I.pct(len(negativas) / len(neg) * 100)}) tienen silueta negativa: están más cerca, en promedio,
de otro grupo que del suyo. Son casos de frontera, y su asignación no debe leerse como una etiqueta firme.</p>
""")
with st.expander("Ver las observaciones con silueta negativa"):
    st.dataframe(I.tabla(negativas[["pais", "anio", "g", "s"]].sort_values("s").rename(columns={"pais": "País", "anio": "Año", "g": "Grupo",
                 "s": "Silueta"}), 3), hide_index=True, width="stretch")

st.subheader("Centroides en el espacio de los componentes")
cent = pd.DataFrame(t["km"].centroides, columns=[f"CP{i + 1}" for i in range(t["nc"])], index=[nombres[g] for g in range(k)])
st.dataframe(I.tabla(cent, 2), width="stretch")
I.fuente("Coordenadas promedio de cada grupo. Signo positivo en CP1: Estado más grande; en CP2: más holgura externa y laboral.")
