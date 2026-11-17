# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st

from regimenes import fuente as F
from regimenes import graficos as G
from regimenes import interfaz as I

I.cabeza("Paso 1 · Exploración", "Qué trae la base",
         "Estructura del archivo, cobertura por año y variable, estadísticas descriptivas, valores extremos y relaciones entre "
         "las variables antes de cualquier transformación.")
I.linea_estado()

pan = I.panel()
q = F.resumen_calidad(pan)
t = I.todo()
prep = t["prep"]
vars_ = prep.variables

st.subheader("Estructura del archivo")
I.datos_clave([
    (I.ent(q["filas"]), "filas, una por país y año"),
    (str(q["paises"]), "economías, sin agregados regionales ni de ingreso"),
    (f"{q['anio_min']}-{q['anio_max']}", "años en el archivo"),
    (str(q["indicadores"]), "indicadores numéricos"),
    (I.pct(q["celdas_vacias"]), "de las celdas de indicadores están vacías"),
])
st.write("")
I.texto(f"""
<p>El archivo tiene tres columnas de identificación (nombre del país, código ISO de dos letras y año) y trece indicadores numéricos,
todos de tipo decimal. No hay filas repetidas para un mismo país y año ({q['duplicados']} duplicados) ni filas de agregados como
«Mundo» o «América Latina». A la base se le agregaron el nombre en español, la región y el nivel de ingreso desde la API del Banco
Mundial; esas columnas sirven para describir resultados y no entran al modelo.</p>
""")
with st.expander("Ver las primeras filas del archivo tal como viene"):
    st.dataframe(F.cargar_base().head(20), hide_index=True, width="stretch")

st.subheader("Cobertura: qué hay y qué falta")
cob = F.cobertura(pan)
st.plotly_chart(G.cobertura(cob), width="stretch")
I.fuente("Porcentaje de las 217 economías con dato en cada año. Banco Mundial, WDI.")
p23 = cob.loc[["ingreso_publico", "deuda_publica", "tasa_real"], 2023]
I.texto(f"""
<p>La cobertura decide buena parte del diseño. El PIB, el crecimiento y la inflación superan el 90 % hasta 2022, pero en 2024 casi todo
desaparece y 2025 no tiene un solo dato, así que la ventana útil termina en 2023. Las cifras fiscales son el cuello de botella: los
ingresos del gobierno central tienen dato para el {I.pct(cob.loc['ingreso_publico', 2019], 0)} de los países en 2019 y para el
{I.pct(p23['ingreso_publico'], 0)} en 2023. La tasa real llega al {I.pct(p23['tasa_real'], 0)} en 2023 y la deuda pública al
{I.pct(p23['deuda_publica'], 0)}, por lo que la primera queda como opcional y la segunda fuera del modelo.</p>
""")

st.subheader("Tipos de variable y papel en el estudio")
filas = []
for v in F.INDICADORES:
    info = F.VARIABLES[v]
    ventana = pan[pan["anio"].between(2010, 2023)][v]
    filas.append({"Variable": info["etiqueta"], "Unidad": info["unidad"], "Tipo": str(pan[v].dtype),
                  "Presente 2010-2023": I.pct(ventana.notna().mean() * 100),
                  "Papel": {"modelo": "Modelo", "opcional": "Opcional", "descartada": "Descartada", "descriptiva": "Descriptiva"}[info["papel"]]})
st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")
I.texto("""
<p>El PIB y el ingreso nacional en dólares miden el tamaño de la economía y tienen una correlación de 1,0 entre sí; incluirlos haría
que Estados Unidos y China se separaran del resto por ser grandes, no por su situación macroeconómica. El PIB per cápita se usa solo
para describir los grupos. El deflactor repite la información de la inflación al consumidor y el recaudo la de los ingresos del
gobierno, así que quedan como alternativas opcionales.</p>
""")

st.subheader("Estadísticas descriptivas del panel del modelo")
desc = prep.datos[vars_].describe(percentiles=[0.25, 0.5, 0.75]).T
desc.index = [F.etiqueta(v) for v in desc.index]
desc.columns = ["n", "Media", "Desv. est.", "Mín.", "P25", "Mediana", "P75", "Máx."]
st.dataframe(I.tabla(desc, 1), width="stretch")
I.fuente("Valores originales, antes de transformar o estandarizar, en las observaciones que entran al modelo vigente.")

d = prep.datos
inf_max = d.loc[d["inflacion"].idxmax()]
cre_min = d.loc[d["crecimiento"].idxmin()]
cre_max = d.loc[d["crecimiento"].idxmax()]
I.texto(f"""
<p>Las escalas son muy distintas. La inflación tiene una mediana de {I.pct(d['inflacion'].median())} pero llega a
{I.pct(inf_max['inflacion'], 0)} en {inf_max['pais']} ({inf_max['anio']}); una cola así dominaría cualquier distancia si no se
transforma. El crecimiento va de {I.pct(cre_min['crecimiento'])} en {cre_min['pais']} ({cre_min['anio']}) a
{I.pct(cre_max['crecimiento'])} en {cre_max['pais']} ({cre_max['anio']}); son caídas y rebotes reales de la pandemia, no errores, y
por eso se conservan.</p>
""")

st.subheader("Distribuciones")
foco = I.estado()["pais"]
st.plotly_chart(G.distribuciones(prep.datos, vars_, foco, I.pais(foco)), width="stretch")

st.subheader("Relaciones entre las variables")
st.plotly_chart(G.correlaciones(prep.X), width="stretch")
c = prep.X.corr()
pares = [(c.iloc[i, j], c.index[i], c.columns[j]) for i in range(len(c)) for j in range(i + 1, len(c))]
pares.sort(key=lambda x: -abs(x[0]))
top = pares[0]
I.texto(f"""
<p>La correlación más fuerte es la de {F.etiqueta(top[1]).lower()} con {F.etiqueta(top[2]).lower()} ({I.num(top[0], 2)}). Le siguen
{F.etiqueta(pares[1][1]).lower()} con {F.etiqueta(pares[1][2]).lower()} ({I.num(pares[1][0], 2)}) y
{F.etiqueta(pares[2][1]).lower()} con {F.etiqueta(pares[2][2]).lower()} ({I.num(pares[2][0], 2)}). El resto de pares queda por debajo de
{I.num(abs(pares[3][0]), 2)} en valor absoluto: hay información común, que el ACP puede resumir, pero ninguna variable es copia de otra.
La matriz usa las variables ya transformadas que entran al modelo.</p>
""")

st.subheader("Valores que no corresponden a una economía típica")
anom = prep.anomalas
if len(anom):
    st.dataframe(I.tabla(anom[["pais", "anio", "ingreso_publico", "gasto_publico", "cuenta_corriente"]].rename(columns={
        "pais": "País", "anio": "Año", "ingreso_publico": "Ingreso público (% PIB)", "gasto_publico": "Gasto público (% PIB)",
        "cuenta_corriente": "Cuenta corriente (% PIB)"}), 1), hide_index=True, width="stretch")
I.texto("""
<p>Timor-Leste reporta ingresos del gobierno de hasta 345 % del PIB y superávits de cuenta corriente de más de 200 % del PIB entre 2010
y 2014. Lo más probable es que reflejen ingresos petroleros muy grandes frente a una economía no petrolera pequeña; en cualquier caso, no
describen una situación macroeconómica comparable con la del resto. La vista de Preparación explica la regla que las aparta y permite probar
alternativas.</p>
""")
