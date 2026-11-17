# -*- coding: utf-8 -*-
from dataclasses import replace

import numpy as np
import pandas as pd
import streamlit as st

from regimenes import fuente as F
from regimenes import interfaz as I
from regimenes import modelo as M
from regimenes import preparacion as P
from regimenes import robustez as R

I.cabeza("Paso 7 · Discusión y conclusiones", "Qué aprendimos del panel",
         "Respuestas a la pregunta del trabajo, una prueba de robustez que se recalcula con la configuración vigente y los límites "
         "de los resultados.")
I.linea_estado()

t = I.todo()
d, et, S = t["prep"].datos, t["km"].etiquetas, t["S"]
nombres, perf, r = t["nombres"], t["perf"], t["acp"]
conteo, prob = M.transiciones(d, et)
quedan = float(np.trace(conteo.to_numpy()) / conteo.to_numpy().sum() * 100)
cambios_pais = pd.Series({i: M.cambios(M.trayectoria(d, et, S, i)) for i in d["iso2"].unique()})
nunca = float((cambios_pais == 0).mean() * 100)
sil = t["km"].silueta_obs
co = M.trayectoria(d, et, S, "co") if (d["iso2"] == "co").any() else None


@st.cache_data(show_spinner="Recalculando la prueba de robustez...")
def robustez(a: P.Ajustes, nc: int, k: int) -> pd.DataFrame:
    return R.tabla_robustez(I.panel(), a, nc, k)


st.subheader("1. Qué dimensiones resumen la macroeconomía del panel")
if I.prep_es_estudio():
    I.texto(f"""
<p>Tres componentes conservan el {I.pct(r.acumulada[2] * 100)} de la variación de seis indicadores. El primero ({I.pct(r.varianza[0] * 100)}) es el
tamaño del Estado: ingresos y gasto del gobierno con cargas de {I.num(r.cargas.loc['ingreso_publico', 'CP1'], 2)} y
{I.num(r.cargas.loc['gasto_publico', 'CP1'], 2)}. El segundo ({I.pct(r.varianza[1] * 100)}) opone la holgura externa y laboral, superávit y
desempleo bajo, al déficit con desempleo alto. El tercero ({I.pct(r.varianza[2] * 100)}) es la presión de precios. Que el eje más importante
sea fiscal, una característica que cambia despacio, ya inclina la respuesta hacia la estructura.</p>
""")
else:
    I.texto(f"<p>Con la configuración vigente, {t['nc']} componentes conservan el {I.pct(r.acumulada[t['nc'] - 1] * 100)} de la variación.</p>")

st.subheader("2. Qué grupos aparecen y cuán nítidos son")
I.texto(f"""
<p>K-Means forma {t['k']} grupos: {", ".join(f"«{n}» ({I.ent(perf['tam'].loc[g])} observaciones)" for g, n in sorted(nombres.items()))}.
La silueta promedio es {I.num(t['km'].silueta, 3)}, en la franja de estructura débil, y el {I.pct((sil < 0).mean() * 100)} de las
observaciones tiene silueta negativa. Las diferencias entre grupos son reales en las medianas, pero las fronteras son difusas: el panel es
una nube continua con zonas densas, no un conjunto de islas.</p>
""")

st.subheader("3. ¿Ciclo o estructura?")
rep = M.reparto_por_anio(d, et)
baja = (rep.loc[2020] - rep.drop(index=2020).mean()).idxmin() if 2020 in rep.index else rep.columns[0]
I.texto(f"""
<p>Estructura, con una excepción. En el {I.pct(quedan)} de los pares de años consecutivos el país se queda en su grupo, y el
{I.pct(nunca, 0)} de los países nunca cambia en todo el periodo. El reparto de los grupos casi no se mueve de un año a otro salvo en 2020,
cuando «{nombres[baja]}» pasa de reunir el {I.pct(rep.drop(index=2020)[baja].mean(), 0)} de las observaciones a reunir el
{I.pct(rep.loc[2020, baja], 0)}. Los grupos describen sobre todo tipos de economía; el ciclo aparece como un desplazamiento dentro de cada
grupo y, en los años de choque, como un cruce de fronteras.</p>
""")
if co is not None and len(co):
    moda = co["grupo"].mode().iloc[0]
    I.texto(f"""
<p>Colombia lo ilustra bien. Pasa {int((co['grupo'] == moda).sum())} de sus {len(co)} años en «{nombres[moda]}»
{"y nunca cambia de grupo" if M.cambios(co) == 0 else f"y cambia de grupo {M.cambios(co)} {'vez' if M.cambios(co) == 1 else 'veces'}"}. Sus vecinos más cercanos en el periodo no son los socios de la Alianza del Pacífico sino economías con el mismo
problema de fondo, desempleo alto y déficit externo: {", ".join(M.gemelos_periodo(d, S, 'co')['pais'].head(4))}.</p>
""")

st.subheader("4. ¿Resiste el resultado a otras decisiones?")
tab = robustez(t["a"], t["nc"], t["k"])
st.dataframe(I.tabla(tab, 1, {"Observaciones": 0, "Silueta": 3, "ARI": 2, "Cambios COL": 0}), hide_index=True, width="stretch")
I.fuente("Cada fila cambia una sola decisión respecto de la configuración vigente. ARI: índice de Rand ajustado entre esa partición y la vigente en "
         "las observaciones comunes (1 = idénticas, 0 = coincidencia al azar). Permanencia: pares de años consecutivos en que el país sigue en el mismo "
         "grupo. Sin cambios: países que nunca cambian de grupo. Cambios COL: veces que Colombia cambia de grupo. COL-BRA, COL-MEX y COL-PER: "
         "porcentaje de los años comunes en que Colombia comparte grupo con Brasil, México y Perú.")
def minus(s: str) -> str:
    return s if s[:1] == "K" else s[:1].lower() + s[1:]


def lista(xs) -> str:
    xs = list(xs)
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " y " + xs[-1]


niveles = tab[~tab["Alternativa"].str.startswith("Desviaciones")]
desv = tab[tab["Alternativa"].str.startswith("Desviaciones")]
otras = niveles.iloc[1:]
if len(otras):
    firme = otras[otras["COL-BRA (%)"] >= 80]
    floja = otras[otras["COL-BRA (%)"] < 80]
    peor, mejor = otras.loc[otras["ARI"].idxmin()], otras.loc[otras["ARI"].idxmax()]
    txt = (f"<p>La respuesta general resiste. En todas las alternativas que comparan a los países en niveles, entre el "
           f"{I.pct(niveles['Permanencia (%)'].min(), 0)} y el {I.pct(niveles['Permanencia (%)'].max(), 0)} de los pares de "
           f"años consecutivos se quedan en el mismo grupo.")
    if len(desv):
        txt += (f" Cuando a cada país se le resta su promedio, la permanencia cae al {I.pct(desv['Permanencia (%)'].iloc[0], 0)} y la "
                f"partición no se parece a la vigente (ARI de {I.num(desv['ARI'].iloc[0], 2)}): el ciclo existe en los datos, "
                f"pero queda en segundo plano cuando se compara a los países entre sí.")
    txt += (f"</p><p>Los grupos concretos, en cambio, dependen de las decisiones. El ARI con la configuración vigente va de "
            f"{I.num(peor['ARI'], 2)} ({minus(peor['Alternativa'])}) a {I.num(mejor['ARI'], 2)} "
            f"({minus(mejor['Alternativa'])}), una coincidencia moderada.")
    if len(firme) and len(floja):
        txt += (f" También el lugar de Colombia: con {lista(minus(a) for a in firme['Alternativa'])} sigue junto a Brasil en al menos el 80 % "
                f"de los años; con {lista(minus(a) for a in floja['Alternativa'])} comparte grupo con Brasil en menos años y cambia de grupo "
                f"entre {int(floja['Cambios COL'].min())} y {int(floja['Cambios COL'].max())} veces.")
        acerca = floja[floja[["COL-MEX (%)", "COL-PER (%)"]].max(axis=1) >= 50]
        if len(acerca):
            txt += f" Con {lista(minus(a) for a in acerca['Alternativa'])} queda la mayoría de los años junto a México o Perú."
        txt += (" Colombia está cerca de la frontera entre el grupo de desempleo alto y el de Estado pequeño, y su etiqueta debe leerse "
                "con esa cautela.")
    txt += "</p>"
    I.texto(txt)


st.subheader("5. Para qué sirve y qué hay que tener en cuenta")
I.texto("""
<p>El ejercicio sirve para elegir con quién compararse. Si Colombia quiere aprender de economías con su mismo problema macroeconómico,
la lista de países con desempleo alto y déficit externo es más útil que la de vecinos geográficos o socios comerciales. También sirve
para leer un choque: 2020 sacó de su grupo a muchas economías de Estado pequeño, pero no cambió el diagnóstico de fondo de Colombia.</p>
<p>Los límites son concretos. Las cifras fiscales cubren solo al gobierno central y faltan para muchos países de ingreso bajo, que por
eso quedan fuera. Lo primero explica por qué Estados Unidos cae en el grupo de Estado pequeño: en un país federal buena parte del gasto
lo hacen los estados, y esa parte no aparece en la base. El desempleo es una estimación modelada de la OIT y no capta la informalidad. Las observaciones de un mismo país no son
independientes, de modo que la persistencia alta es en parte esperable y debe leerse como descripción, no como prueba estadística. Y la
silueta débil obliga a tratar cada asignación como un perfil aproximado, sobre todo en los casos de frontera.</p>
""")

st.subheader("Conclusión")
if I.es_estudio():
    I.aviso("""Entre 2010 y 2023, lo que más separa a las economías del mundo en estos seis indicadores es su estructura, sobre todo el tamaño del
Estado y la combinación de desempleo y cuenta corriente, más que el momento del ciclo: casi nueve de cada diez veces un país sigue al año
siguiente en el mismo grupo. En la configuración del estudio, Colombia pasa los catorce años en el grupo de desempleo alto y déficit externo,
sin que la caída del petróleo ni la pandemia la saquen de él, aunque su ubicación cerca de la frontera la hace sensible a algunas decisiones
de preparación.""")
else:
    I.aviso("La conclusión del estudio corresponde a su configuración. Con la vigente, revisa las secciones anteriores y la tabla de "
            "robustez, o vuelve a la configuración del estudio.")
