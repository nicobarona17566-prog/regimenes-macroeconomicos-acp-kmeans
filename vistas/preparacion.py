# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import streamlit as st

from regimenes import fuente as F
from regimenes import graficos as G
from regimenes import interfaz as I
from regimenes import preparacion as P

I.cabeza("Paso 2 · Preparación", "Decisiones antes de modelar",
         "Qué observaciones y variables entran, qué se hace con los valores anómalos y por qué todo se estandariza. "
         "Cada decisión se puede cambiar aquí y el resto de la aplicación se recalcula.")

with st.container(border=True):
    st.markdown("**Decisiones que puedes cambiar**")
    c1, c2 = st.columns(2)
    with c1:
        I.control_ventana("Ventana de años", min_value=2010, max_value=2025, step=1,
                          help="Los datos útiles terminan en 2023. Incluir 2024 o 2025 no agrega observaciones completas.")
        I.control(st.multiselect, "variables", "Variables del modelo", options=F.BASE + F.OPCIONALES, format_func=F.etiqueta,
                  help="Al menos cuatro. Agregar la tasa real o el recaudo reduce la muestra porque tienen más vacíos.")
    with c2:
        I.control(st.radio, "atipicos", "Valores anómalos", options=["excluir", "winsorizar", "ninguno"], horizontal=True,
                  format_func={"excluir": "Excluir por regla", "winsorizar": "Winsorizar 1 %-99 %", "ninguno": "Dejarlos"}.get)
        I.control(st.toggle, "log_inflacion", "Logaritmo con signo en la inflación")
        I.control(st.radio, "enfoque", "Enfoque", options=["niveles", "desviaciones"], horizontal=True,
                  format_func={"niveles": "Niveles", "desviaciones": "Desviación del promedio de cada país"}.get,
                  help="Niveles compara a los países entre sí. Desviaciones quita a cada país su promedio y deja solo el ciclo.")
    if len(I.estado()["variables"]) < 4:
        st.warning("El modelo necesita al menos cuatro variables. Mientras tanto se usan las seis del estudio.")

I.linea_estado()
t = I.todo()
prep = t["prep"]
a = t["a"]
pan = I.panel()

st.subheader("Del archivo al panel del modelo")
et = pd.DataFrame(prep.etapas, columns=["Paso", "Observaciones", "Países"])
et["Quedan"] = (et["Observaciones"] / et["Observaciones"].iloc[0] * 100).map(lambda v: I.pct(v, 0))
st.dataframe(I.tabla(et, 0), hide_index=True, width="stretch")

st.subheader("El costo de cada variable en observaciones")
cand = F.BASE + F.OPCIONALES + ["deuda_publica"]
fv = P.faltantes_por_variable(pan, a, cand)
fv["Variable"] = fv["variable"].map(F.etiqueta)
fv["Estado"] = np.where(fv["en_modelo"], "En el modelo", "Si se agrega")
fv = fv.rename(columns={"observaciones": "Observaciones completas", "paises": "Países", "presentes": "% presente en la ventana"})
st.dataframe(I.tabla(fv[["Variable", "Estado", "Observaciones completas", "Países", "% presente en la ventana"]], 1),
             hide_index=True, width="stretch")
I.fuente("Para las variables del modelo, la fila muestra el panel actual. Para las demás, cuántas observaciones quedarían si se agregaran.")

st.subheader("Las decisiones del estudio")
sin_tasa = fv.loc[fv["variable"] == "tasa_real"]
tasa_obs = int(sin_tasa["Observaciones completas"].iloc[0]) if len(sin_tasa) else 0
tasa_pai = int(sin_tasa["Países"].iloc[0]) if len(sin_tasa) else 0
base_prep = I.preparacion(P.ESTUDIO)
I.texto(f"""
<p>1. Unidad de observación: un país en un año. Se descarta trabajar con un solo país porque 14 años no alcanzan para estimar
componentes y grupos con seis variables, y se descarta el promedio por país porque borraría el tiempo, que es la mitad de la
pregunta.</p>
<p>2. Ventana 2010-2023. El archivo llega a 2025, pero en 2024 solo hay inflación y desempleo para parte de los países y en 2025 no hay
datos. Extender la ventana no suma observaciones completas.</p>
<p>3. Seis variables que cubren precios (inflación), actividad (crecimiento), empleo (desempleo), sector externo (cuenta corriente) y
política fiscal (ingreso y gasto del gobierno). Quedan fuera el PIB y el ingreso nacional, que miden tamaño; el PIB per cápita, que se
reserva para describir los grupos; la deuda pública, presente en apenas una cuarta parte del panel; y el deflactor y el recaudo, que
repiten información de la inflación y del ingreso público. La tasa real es la candidata más tentadora, pero agregarla deja
{I.ent(tasa_obs)} observaciones de {tasa_pai} países.</p>
<p>4. Casos completos, sin imputación. Rellenar datos fiscales que el país no reportó con promedios o modelos inventaría justo la
información que separa a los grupos. El costo es que los países sin estadísticas fiscales, muchos de ingreso bajo, quedan fuera.</p>
<p>5. Regla de anómalos: se apartan las observaciones con ingresos o gasto del gobierno por encima del {I.ent(P.TOPE_FISCAL)} % del PIB
o con un saldo de cuenta corriente mayor al {I.ent(P.TOPE_EXTERNO)} % del PIB en valor absoluto. Con las variables del estudio la regla
aparta {len(base_prep.anomalas)} observaciones, todas de Timor-Leste. Las caídas y rebotes de la pandemia, como los de Macao o Maldivas,
se conservan porque son hechos reales.</p>
<p>6. Logaritmo con signo en la inflación: ln(1 + |x|) con el signo de x. Una inflación de 3 % pasa a {I.num(np.log1p(3), 2)}, una de
557 % a {I.num(np.log1p(557), 2)} y una deflación de 2 % a {I.num(-np.log1p(2), 2)}. La transformación conserva el orden y el signo, pero
impide que tres o cuatro hiperinflaciones definan por sí solas un componente.</p>
<p>7. Estandarización: cada variable se lleva a media cero y desviación uno. El desempleo se mide en puntos de la fuerza laboral y el
gasto público en puntos del PIB; sin estandarizar, la variable con más dispersión dominaría las distancias de K-Means y la varianza del
ACP.</p>
""")

st.subheader("Antes y después de estandarizar")
c1, c2 = st.columns(2)
c1.plotly_chart(G.cajas_escala(prep.X, "Variables transformadas, en sus unidades"), width="stretch")
c2.plotly_chart(G.cajas_escala(pd.DataFrame(prep.Z, columns=prep.variables), "Variables estandarizadas"), width="stretch")
I.texto("""
<p>A la izquierda, cada variable conserva su escala: el gasto público se mueve entre 10 % y 50 % del PIB y la inflación transformada entre
-1 y 3. A la derecha todas tienen media cero y desviación uno, y lo que queda a la vista son las colas: los puntos alejados son
observaciones extremas que el ACP y K-Means van a tratar como tales.</p>
""")

if a.enfoque == "desviaciones":
    I.aviso("Con el enfoque de desviaciones, a cada variable se le resta el promedio del país antes de estandarizar. El modelo ya no "
            "compara a Colombia con Alemania sino a Colombia en 2020 con Colombia en su año típico. Los grupos que salen así describen "
            "momentos del ciclo, y por eso no reciben los nombres del estudio.")
if a.atipicos == "winsorizar" and prep.recortes:
    lim = pd.DataFrame(prep.recortes, index=["Límite inferior", "Límite superior"]).T
    lim.index = [F.etiqueta(v) for v in lim.index]
    st.markdown("Límites de la winsorización (percentiles 1 y 99, en la escala transformada)")
    st.dataframe(I.tabla(lim, 2), width="stretch")

st.subheader("Matriz que entra al modelo")
salida = pd.concat([prep.datos[["iso3", "pais", "anio"]], pd.DataFrame(prep.Z, columns=[f"z_{v}" for v in prep.variables])], axis=1)
st.dataframe(I.tabla(salida.head(12), 3), hide_index=True, width="stretch")
st.download_button("Descargar la matriz estandarizada (CSV)", salida.to_csv(index=False).encode("utf-8"),
                   file_name="matriz_estandarizada.csv", mime="text/csv")
