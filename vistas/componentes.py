# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import streamlit as st

from regimenes import fuente as F
from regimenes import graficos as G
from regimenes import interfaz as I
from regimenes import modelo as M

I.cabeza("Paso 3 · Análisis de componentes principales", "Tres ejes para seis variables",
         "Cuánta información conserva cada componente, cuántos se retienen y qué significa cada uno en términos macroeconómicos.")

with st.expander("Cómo funciona el ACP, en pocas líneas"):
    I.texto("""
<p>Con la matriz estandarizada Z, el ACP calcula la matriz de correlaciones R y sus vectores y valores propios. Cada vector propio define
un componente: una combinación lineal de las variables originales, sin correlación con las demás combinaciones. El valor propio λ de un
componente es la varianza que recoge; como cada variable estandarizada aporta una unidad de varianza, el total es igual al número de
variables y λ / p es la proporción explicada.</p>
<p>La carga de una variable en un componente es su correlación con ese componente: el elemento del vector propio multiplicado por la raíz
de λ. Una carga cercana a 1 o a -1 indica que la variable define el eje; una cercana a 0, que el eje no la recoge. El signo de un
componente es arbitrario, así que aquí se orienta para que la variable con la carga más grande quede positiva.</p>
""")

t0 = I.todo()
p = len(t0["prep"].variables)
r = t0["acp"]
sug = M.sugerir_componentes(r)

st.subheader("Varianza explicada")
c1, c2 = st.columns(2)
c1.plotly_chart(G.sedimentacion(r.valores_propios), width="stretch")
with c2:
    I.control(st.slider, "n_comp", "Componentes que se conservan", min_value=2, max_value=p, step=1,
              help="El número de componentes define el espacio en el que trabaja K-Means.")
t = I.todo()
nc = t["nc"]
with c2:
    st.plotly_chart(G.varianza(r.varianza, r.acumulada, nc), width="stretch")
I.linea_estado()

I.datos_clave([
    (str(sug["kaiser"]), f"componentes con valor propio mayor que 1 (criterio de Kaiser), {I.pct(r.acumulada[sug['kaiser'] - 1] * 100)}"),
    (str(sug["acumulada"]), f"componentes para superar el 70 % de la varianza, {I.pct(r.acumulada[sug['acumulada'] - 1] * 100)}"),
    (I.num(r.valores_propios[0] - r.valores_propios[1], 2), "caída entre el primer y el segundo valor propio, la única brusca del gráfico"),
    (str(nc), f"componentes conservados, {I.pct(r.acumulada[nc - 1] * 100)} de la información"),
])
st.write("")

if I.prep_es_estudio():
    I.texto(f"""
<p>Los tres criterios no coinciden, y la decisión tiene que tomar partido. El criterio de Kaiser se detiene en dos componentes, que
juntos conservan el {I.pct(r.acumulada[1] * 100)} de la varianza. El gráfico de sedimentación no muestra un codo útil: la única caída
brusca está entre el primer y el segundo valor propio, y quedarse con uno solo dejaría fuera dos tercios de la información. El estudio
conserva tres componentes, que llegan al {I.pct(r.acumulada[2] * 100)} y superan el umbral habitual del 70 %. El tercer valor propio
({I.num(r.valores_propios[2], 2)}) está por debajo de 1, pero muy cerca, y el componente tiene una lectura económica clara, la presión de
precios, que los dos primeros no recogen. La frontera con el cuarto es tenue: su valor propio ({I.num(r.valores_propios[3], 2)}) casi
iguala al tercero, y el componente mezcla crecimiento alto con inflación baja, una combinación que los tres primeros ya recogen en parte.
La sección de robustez, en Conclusiones, muestra qué cambia si se conservan cuatro. El sexto componente ({I.pct(r.varianza[5] * 100)})
refleja casi solo la diferencia entre ingreso y gasto público, dos variables que se mueven juntas.</p>
""")

st.subheader("Cargas: qué variables definen cada componente")
st.plotly_chart(G.cargas(r.cargas, nc), width="stretch")
I.fuente("Correlación entre cada variable y cada componente. Azul: relación negativa; rojo: positiva.")

NOMBRES_CP = {1: "Tamaño del Estado", 2: "Holgura externa y laboral", 3: "Presión de precios"}
if I.prep_es_estudio():
    cg = r.cargas
    I.texto(f"""
<p>CP1, tamaño del Estado ({I.pct(r.varianza[0] * 100)}). Lo definen el ingreso público ({I.num(cg.loc['ingreso_publico', 'CP1'], 2)}) y el
gasto público ({I.num(cg.loc['gasto_publico', 'CP1'], 2)}). Lo acompañan, con menos peso, un desempleo más alto
({I.num(cg.loc['desempleo', 'CP1'], 2)}), menos inflación ({I.num(cg.loc['inflacion', 'CP1'], 2)}) y menos crecimiento
({I.num(cg.loc['crecimiento', 'CP1'], 2)}). Es el eje que separa a las economías ricas de Estado grande, con precios estables y
crecimiento moderado, de las economías de Estado pequeño que crecen más rápido.</p>
<p>CP2, holgura externa y laboral ({I.pct(r.varianza[1] * 100)}). Un valor alto combina superávit de cuenta corriente
({I.num(cg.loc['cuenta_corriente', 'CP2'], 2)}), desempleo bajo ({I.num(cg.loc['desempleo', 'CP2'], 2)}) y más crecimiento
({I.num(cg.loc['crecimiento', 'CP2'], 2)}). Un valor bajo es la combinación opuesta, déficit externo con desempleo alto, que es la de
Colombia.</p>
<p>CP3, presión de precios ({I.pct(r.varianza[2] * 100)}). Lo domina la inflación ({I.num(cg.loc['inflacion', 'CP3'], 2)}) junto con el
crecimiento ({I.num(cg.loc['crecimiento', 'CP3'], 2)}): distingue los años en que la economía se calienta, como 2022, de los años con
precios quietos.</p>
""")
else:
    lineas = []
    for i in range(nc):
        col = f"CP{i + 1}"
        top = r.cargas[col].abs().sort_values(ascending=False).index[:3]
        partes = ", ".join(f"{F.etiqueta(v).lower()} ({I.num(r.cargas.loc[v, col], 2)})" for v in top)
        lineas.append(f"<p>{col} ({I.pct(r.varianza[i] * 100)}): {partes}.</p>")
    I.texto("<p>Con esta configuración, las variables con más peso en cada componente son:</p>" + "".join(lineas)
            + "<p>Los nombres que el estudio da a los componentes valen solo para su configuración; aquí conviene leer las cargas.</p>")

st.subheader("Las observaciones en el espacio de los componentes")
c1, c2, c3 = st.columns([1.3, 1, 1])
color_por = c1.radio("Colorear por", ["Periodo", "Región", "Nivel de ingreso"], horizontal=True)
ejes_op = ["CP1 y CP2"] + (["CP1 y CP3", "CP2 y CP3"] if nc >= 3 else [])
ejes_txt = c2.selectbox("Ejes", ejes_op)
biplot = c3.toggle("Mostrar las cargas (biplot)", value=True)
ejes = {"CP1 y CP2": (0, 1), "CP1 y CP3": (0, 2), "CP2 y CP3": (1, 2)}[ejes_txt]

d = t["prep"].datos
if color_por == "Periodo":
    per = pd.cut(d["anio"], [2009, 2014, 2019, 2020, 2025], labels=["2010-2014", "2015-2019", "2020", "2021-2023"]).astype(str)
    cats = ["2010-2014", "2015-2019", "2021-2023", "2020"]
    grupos = per
elif color_por == "Región":
    grupos = d["region"].map(lambda x: F.REGION_CORTA.get(x, x))
    cats = list(grupos.value_counts().index)
else:
    grupos = d["ingreso"]
    cats = [c for c in F.ORDEN_INGRESO if c in set(grupos)]
colores = {c: I.PALETA[i % len(I.PALETA)] for i, c in enumerate(cats)}
if color_por == "Periodo":
    colores = {"2010-2014": "#9FB3C2", "2015-2019": "#5E7A91", "2021-2023": "#B57C00", "2020": "#D1335F"}
foco = I.estado()["pais"]
fig = G.plano(d, r.scores, grupos, colores, None, r.cargas if biplot else None, foco, I.pais(foco),
              (r.varianza[ejes[0]] * 100, r.varianza[ejes[1]] * 100), ejes=ejes)
st.plotly_chart(fig, width="stretch")
I.fuente(f"El plano resume el {I.pct((r.varianza[ejes[0]] + r.varianza[ejes[1]]) * 100)} de la varianza total. La línea negra une los años de "
         f"{I.pais(foco)}; se rotulan el primero, el último y 2020.")
if I.prep_es_estudio() and ejes == (0, 1):
    I.texto("""
<p>Coloreado por periodo, el plano muestra que 2020 se desplaza hacia abajo y a la derecha: más gasto, más desempleo y menos crecimiento.
Coloreado por nivel de ingreso, el primer eje ordena casi solo: los países de ingreso alto se concentran a la derecha. Es una primera
pista para la pregunta del trabajo: el eje que más separa a las observaciones es estructural, no coyuntural.</p>
""")

st.download_button("Descargar las coordenadas en los componentes (CSV)",
                   pd.concat([d[["iso3", "pais", "anio"]], pd.DataFrame(r.scores[:, :nc], columns=[f"CP{i + 1}" for i in range(nc)])], axis=1)
                   .to_csv(index=False).encode("utf-8"), file_name="coordenadas_componentes.csv", mime="text/csv")
