# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st

from regimenes import fuente as F
from regimenes import interfaz as I
from regimenes import preparacion as P

I.cabeza("Contexto y pregunta", "¿Ciclo o estructura?",
         "Qué separa a las economías del mundo entre 2010 y 2023 cuando se miran seis indicadores macroeconómicos, "
         "y en qué grupo queda Colombia año tras año.")

pan = I.panel()
prep = I.preparacion(P.ESTUDIO)
d = prep.datos
co = pan[(pan["iso2"] == "co") & pan["anio"].between(2010, 2023)]

I.datos_clave([
    (I.ent(len(d)), "observaciones país-año en el modelo del estudio"),
    (str(d["iso2"].nunique()), "países con las seis variables en al menos un año"),
    ("6", "indicadores: precios, actividad, empleo, sector externo y fiscal"),
    ("2010-2023", "años con datos; 2024 y 2025 vienen casi vacíos"),
])
st.write("")

I.texto(f"""
<p>Cuando se compara la macroeconomía de dos países se mezclan dos cosas. Una es el momento del ciclo: si el año fue de expansión,
de recesión o de inflación alta. La otra es la estructura: el tamaño del Estado, la posición externa y el funcionamiento del mercado
laboral, rasgos que cambian despacio. Este trabajo pregunta cuál de las dos domina cuando se agrupan los países sin decirle al
algoritmo qué buscar. Si los grupos son estados del ciclo, los países deberían pasar de uno a otro con los años. Si son tipos de
economía, cada país debería quedarse casi siempre en el mismo grupo.</p>
<p>Colombia es el caso que guía la lectura. Entre 2010 y 2023 su desempleo tuvo una mediana de {I.pct(co['desempleo'].median())},
su cuenta corriente un saldo mediano de {I.pct(co['cuenta_corriente'].median())} del PIB y los ingresos del gobierno central
rondaron el {I.pct(co['ingreso_publico'].median(), 0)} del PIB. La pregunta concreta es a qué economías se parece esa combinación
y si la pandemia, la caída del petróleo de 2015 o la inflación de 2022 la sacaron de su grupo.</p>
""")

st.subheader("Qué representa cada observación")
I.texto("""
<p>Cada fila del modelo es un país en un año: Colombia en 2015, Chile en 2020, Alemania en 2012. Trabajar con el panel completo, y no
con un promedio por país, tiene dos ventajas. La primera es el tamaño: un solo país ofrece 14 años, muy pocos para estimar componentes
y grupos con seis variables, mientras que el panel reúne más de mil seiscientas observaciones. La segunda es que permite seguir a cada
país en el tiempo y medir si cambia de grupo, que es justamente lo que separa al ciclo de la estructura.</p>
""")

st.subheader("Por qué ACP y K-Means")
I.texto("""
<p>Los seis indicadores no son independientes. Los ingresos y el gasto del gobierno se mueven juntos, y el desempleo, la inflación
y el crecimiento responden a los mismos choques. El análisis de componentes principales resume esas relaciones en unas pocas
dimensiones y muestra qué combinaciones de variables separan más a las observaciones. Esas dimensiones son la base para dibujar el
panel en un plano y para medir distancias sin que dos variables casi gemelas cuenten doble.</p>
<p>K-Means busca grupos de observaciones con perfiles parecidos sin categorías impuestas de antemano. Aquí sirve para dos cosas:
describir qué tipos de situación macroeconómica aparecen en el mundo y, gracias a la dimensión temporal, contar cuántas veces cada
país cambia de grupo.</p>
""")

st.subheader("La fuente")
I.texto("""
<p>La base es un extracto de los Indicadores del Desarrollo Mundial (WDI) del Banco Mundial con una fila por país y año: 217
economías, de 2010 a 2025, y 13 indicadores. Para comprobar el origen se descargaron de la API del Banco Mundial seis de esos
indicadores para Colombia. En inflación, desempleo, PIB, PIB per cápita, recaudo y tasa real coinciden entre 11 y 14 de los 14 años; las
diferencias, menores al 1 % salvo en la tasa real, corresponden a revisiones que el Banco hizo después de armar el archivo.</p>
""")
cat = pd.DataFrame([dict(Variable=v["etiqueta"], Descripción=v["larga"], Código=v["codigo"],
                         Uso={"modelo": "Entra al modelo", "opcional": "Opcional en la aplicación",
                              "descartada": "No entra", "descriptiva": "Solo para describir grupos"}[v["papel"]])
                    for v in F.VARIABLES.values()])
st.dataframe(cat, hide_index=True, width="stretch")
I.fuente("Banco Mundial, World Development Indicators, licencia CC BY 4.0. El PIB y el ingreso nacional miden tamaño, no situación "
         "macroeconómica; la deuda pública solo tiene el 24,5 % de los datos.")

st.subheader("Limitaciones que conviene tener presentes")
I.texto("""
<p>El archivo se llama 2025, pero los datos útiles terminan en 2023: en 2024 solo hay inflación y desempleo para parte de los países,
y 2025 está vacío. Las cifras fiscales cubren solo al gobierno central, lo que subestima el tamaño del Estado en los países federales,
y faltan para muchos países de ingreso bajo, de modo que el panel no es balanceado y los últimos años tienen menos observaciones. El desempleo es una estimación modelada de la OIT y no
refleja la informalidad, que en Colombia ronda la mitad del empleo según el DANE. Por último, las observaciones de un mismo país no son
independientes entre sí: el modelo las trata como puntos distintos, y por eso la persistencia de un país en su grupo es un resultado
que hay que leer, no un supuesto.</p>
""")

st.subheader("Cómo recorrer la aplicación")
I.texto("""
<p>Cada vista corresponde a un paso del análisis y todos los cálculos se hacen en vivo. En Preparación se pueden cambiar la ventana
de años, las variables, el tratamiento de los valores anómalos y el enfoque. En Componentes se elige cuántos componentes conservar y en
K-Means cuántos grupos formar. Grupos en el tiempo y Trayectorias leen el resultado año por año, y Conclusiones resume lo aprendido
junto con una prueba de robustez que se recalcula con cada cambio. El botón Volver al estudio restablece la configuración del análisis
principal.</p>
""")

with st.expander("Referencias"):
    I.texto("""
<p>Banco Mundial (2025). World Development Indicators. https://databank.worldbank.org/source/world-development-indicators</p>
<p>Hennig, C. (2007). Cluster-wise assessment of cluster stability. Computational Statistics &amp; Data Analysis, 52(1), 258-271.</p>
<p>Hubert, L. y Arabie, P. (1985). Comparing partitions. Journal of Classification, 2(1), 193-218.</p>
<p>Jolliffe, I. T. y Cadima, J. (2016). Principal component analysis: a review and recent developments. Philosophical Transactions of
the Royal Society A, 374(2065), 20150202.</p>
<p>Kaiser, H. F. (1960). The application of electronic computers to factor analysis. Educational and Psychological Measurement, 20(1),
141-151.</p>
<p>Kaufman, L. y Rousseeuw, P. J. (1990). Finding Groups in Data: An Introduction to Cluster Analysis. Wiley.</p>
<p>Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. Journal of
Computational and Applied Mathematics, 20, 53-65.</p>
""")
