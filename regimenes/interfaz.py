# -*- coding: utf-8 -*-
"""Piezas de la aplicación: estado compartido entre vistas, cálculos con caché, tema y formatos."""
from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import streamlit as st

from . import fuente as F
from . import modelo as M
from . import preparacion as P

N_COMP_ESTUDIO = 3
K_ESTUDIO = 3
PAIS_ESTUDIO = "co"

DEFECTOS = dict(asdict(P.ESTUDIO), n_comp=N_COMP_ESTUDIO, k=K_ESTUDIO, pais="co", comparar=["cl", "br", "mx"])
DEFECTOS["variables"] = list(DEFECTOS["variables"])

# ------------------------------------------------------------------ colores
FONDO, PANEL, TINTA, TENUE, LINEA, ACENTO = "#F4F6F8", "#FFFFFF", "#13202E", "#5B6875", "#DDE3E9", "#0B5F66"
PALETA = ["#B57C00", "#2F6FD0", "#D1335F", "#00998A", "#7B55D0", "#6E8F14", "#8C6D1F", "#3E4C59"]
SIMBOLOS = ["circle", "square", "diamond", "triangle-up", "x", "star", "hexagon", "cross"]
COLOR_NOMBRE = {M.NOMBRES["pequeno"]: PALETA[0], M.NOMBRES["grande"]: PALETA[1],
                M.NOMBRES["fragil"]: PALETA[2], M.NOMBRES["recesion"]: PALETA[4]}
SIMBOLO_NOMBRE = {M.NOMBRES["pequeno"]: "circle", M.NOMBRES["grande"]: "square",
                  M.NOMBRES["fragil"]: "diamond", M.NOMBRES["recesion"]: "x"}


# ------------------------------------------------------------------ estado
def iniciar() -> None:
    s = st.session_state
    if "ajustes" not in s:
        s["ajustes"] = {k: (list(v) if isinstance(v, list) else v) for k, v in DEFECTOS.items()}


def estado() -> dict:
    iniciar()
    return st.session_state["ajustes"]


def control(fn, clave: str, *args, **kwargs):
    """Widget cuyo valor vive en el estado compartido y sobrevive al cambio de vista."""
    k = f"w_{clave}"
    if k not in st.session_state:
        st.session_state[k] = estado()[clave]
    valor = fn(*args, key=k, **kwargs)
    estado()[clave] = list(valor) if isinstance(valor, (list, tuple)) else valor
    return valor


def control_ventana(*args, **kwargs):
    """Deslizador de rango para los años (dos claves del estado en un solo widget)."""
    e = estado()
    if "w_ventana" not in st.session_state:
        st.session_state["w_ventana"] = (int(e["anio_inicio"]), int(e["anio_fin"]))
    a0, a1 = st.slider(*args, key="w_ventana", **kwargs)
    e["anio_inicio"], e["anio_fin"] = int(a0), int(a1)
    return a0, a1


def restablecer() -> None:
    st.session_state["ajustes"] = {k: (list(v) if isinstance(v, list) else v) for k, v in DEFECTOS.items()}
    for k in [k for k in st.session_state if str(k).startswith("w_")]:
        del st.session_state[k]


def ajustes() -> P.Ajustes:
    e = estado()
    vars_ = [v for v in F.BASE + F.OPCIONALES if v in e["variables"]]
    if len(vars_) < 4:
        vars_ = list(F.BASE)
    return P.Ajustes(int(e["anio_inicio"]), int(e["anio_fin"]), tuple(vars_), e["atipicos"], bool(e["log_inflacion"]), e["enfoque"])


def es_estudio() -> bool:
    e = estado()
    return ajustes() == P.ESTUDIO and int(e["n_comp"]) == N_COMP_ESTUDIO and int(e["k"]) == K_ESTUDIO


def prep_es_estudio() -> bool:
    return ajustes() == P.ESTUDIO


def nombrables() -> bool:
    a = ajustes()
    return a.enfoque == "niveles" and set(F.BASE) <= set(a.variables)


# ------------------------------------------------------------------ cálculos con caché
@st.cache_data(show_spinner=False)
def panel() -> pd.DataFrame:
    return F.panel()


@st.cache_data(show_spinner="Preparando el panel...")
def preparacion(a: P.Ajustes) -> P.Preparacion:
    return P.preparar(panel(), a)


@st.cache_data(show_spinner="Calculando componentes principales...")
def acp(a: P.Ajustes) -> M.ResultadoACP:
    p = preparacion(a)
    return M.acp(p.Z, p.variables)


@st.cache_data(show_spinner="Comparando valores de K (silueta, codo y estabilidad)...")
def tabla_k(a: P.Ajustes, n_comp: int) -> pd.DataFrame:
    return M.evaluar_k(acp(a).coordenadas(n_comp))


@st.cache_data(show_spinner="Agrupando con K-Means...")
def kmeans(a: P.Ajustes, n_comp: int, k: int) -> M.ResultadoKMeans:
    return M.segmentar(acp(a).coordenadas(n_comp), k)


def n_comp_valido() -> int:
    p = len(ajustes().variables)
    return int(min(max(2, estado()["n_comp"]), p))


def todo() -> dict:
    """Preparación, ACP, K-Means, perfiles y nombres con la configuración vigente."""
    a = ajustes()
    nc = n_comp_valido()
    k = int(estado()["k"])
    prep = preparacion(a)
    r = acp(a)
    km = kmeans(a, nc, k)
    perf = M.perfiles(prep.datos, km.etiquetas, prep.variables)
    nombres = M.nombres_grupos(perf, nombrables())
    return dict(a=a, nc=nc, k=k, prep=prep, acp=r, S=r.coordenadas(nc), km=km, perf=perf, nombres=nombres,
                colores=colores(nombres), simbolos=simbolos(nombres))


def colores(nombres: dict[int, str]) -> dict[int, str]:
    out = {}
    for g, n in nombres.items():
        out[g] = COLOR_NOMBRE.get(n, PALETA[g % len(PALETA)])
    return out


def simbolos(nombres: dict[int, str]) -> dict[int, str]:
    return {g: SIMBOLO_NOMBRE.get(n, SIMBOLOS[g % len(SIMBOLOS)]) for g, n in nombres.items()}


# ------------------------------------------------------------------ formatos
def num(x, d: int = 1) -> str:
    """Número con coma decimal y punto de miles."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "s. d."
    s = f"{x:,.{d}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def ent(x) -> str:
    return num(x, 0)


def pct(x, d: int = 1) -> str:
    return f"{num(x, d)} %"


def _celda(v, d: int):
    if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)):
        return "-"
    if isinstance(v, (bool, np.bool_)):
        return v
    if isinstance(v, (int, np.integer)):
        return str(v) if 1900 <= v <= 2100 else ent(v)
    if isinstance(v, (float, np.floating)):
        return num(v, d)
    return v


def tabla(df: pd.DataFrame, d: int = 1, cols: dict | None = None):
    """Tabla con coma decimal; los enteros que parecen años quedan sin separador de miles.
    `cols` fija los decimales de columnas concretas."""
    cols = cols or {}
    df = df.copy()
    for c in df.columns:  # Streamlit muestra los vacíos como «None»: se convierten a texto antes de dar formato
        if df[c].dtype.kind in "f" and df[c].isna().any():
            df[c] = df[c].map(lambda v, dd=cols.get(c, d): _celda(v, dd))
    fmt = {c: (lambda v, dd=cols.get(c, d): _celda(v, dd)) for c in df.columns}
    st_ = df.style.format(fmt, na_rep="-")
    return st_.format_index(lambda v: _celda(v, d), axis=0) if df.index.dtype.kind in "if" and not isinstance(df.index, pd.RangeIndex) else st_


def pais(iso2: str) -> str:
    p = panel()
    s = p.loc[p["iso2"] == iso2, "pais"]
    return s.iloc[0] if len(s) else iso2


# ------------------------------------------------------------------ tema
CSS = f"""
<style>
.block-container {{padding-top: 4.2rem; max-width: 1180px;}}
h1, h2, h3 {{letter-spacing: -0.01em;}}
.cabeza {{border-bottom: 1px solid {LINEA}; padding-bottom: .9rem; margin-bottom: 1.1rem;}}
.cabeza .paso {{font-size: .78rem; letter-spacing: .14em; text-transform: uppercase; color: {ACENTO}; font-weight: 600;}}
.cabeza h1 {{font-size: 2.2rem; line-height: 1.1; margin: .3rem 0 .4rem 0; padding: 0;}}
.cabeza p {{color: {TENUE}; font-size: 1.04rem; margin: 0; max-width: 860px;}}
.texto {{font-size: 1rem; line-height: 1.62; color: {TINTA}; max-width: 880px;}}
.texto p {{margin-bottom: .75rem;}}
.dato {{background: {PANEL}; border: 1px solid {LINEA}; border-left: 4px solid {ACENTO}; border-radius: 6px; padding: .8rem 1rem; height: 100%;}}
.dato .v {{font-family: 'Archivo', sans-serif; font-weight: 700; font-size: clamp(1.3rem, 2.1vw, 1.75rem); line-height: 1.1; white-space: nowrap; color: {TINTA};}}
.dato .l {{font-size: .85rem; color: {TENUE}; margin-top: .3rem; line-height: 1.35;}}
.aviso {{background: #EAF3F3; border: 1px solid #C7DFE0; border-radius: 6px; padding: .7rem .95rem; font-size: .93rem; line-height: 1.5; color: {TINTA}; max-width: 900px;}}
.estado {{font-size: .84rem; color: {TENUE}; margin: -.4rem 0 1rem 0;}}
.estado b {{color: {TINTA}; font-weight: 600;}}
.regimen {{background: {PANEL}; border: 1px solid {LINEA}; border-radius: 8px; padding: .85rem 1rem; height: 100%;}}
.regimen h4 {{margin: 0 0 .25rem 0; font-size: 1.05rem; display: flex; align-items: center; gap: .5rem;}}
.regimen .sw {{width: 12px; height: 12px; border-radius: 3px; display: inline-block; flex: none;}}
.regimen .n {{color: {TENUE}; font-size: .84rem;}}
.regimen p {{font-size: .91rem; line-height: 1.45; margin: .45rem 0 0 0;}}
.fuente {{color: #7A8692; font-size: .8rem; margin-top: -.3rem;}}
</style>
"""


def estilo() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def cabeza(paso: str, titulo: str, bajada: str) -> None:
    st.markdown(f'<div class="cabeza"><div class="paso">{paso}</div><h1>{titulo}</h1><p>{bajada}</p></div>', unsafe_allow_html=True)


def texto(html: str) -> None:
    st.markdown(f'<div class="texto">{html}</div>', unsafe_allow_html=True)


def aviso(html: str) -> None:
    st.markdown(f'<div class="aviso">{html}</div>', unsafe_allow_html=True)


def fuente(txt: str) -> None:
    st.markdown(f'<div class="fuente">{txt}</div>', unsafe_allow_html=True)


def datos_clave(items: list[tuple[str, str]]) -> None:
    cols = st.columns(len(items))
    for c, (v, l) in zip(cols, items):
        c.markdown(f'<div class="dato"><div class="v">{v}</div><div class="l">{l}</div></div>', unsafe_allow_html=True)


def linea_estado() -> None:
    """Resumen de la configuración vigente, con un botón para volver a la del estudio."""
    e = estado()
    a = ajustes()
    t = todo()
    atip = {"excluir": "anómalos excluidos", "winsorizar": "winsorizado 1-99", "ninguno": "sin tratar atípicos"}[a.atipicos]
    enfoque = "niveles" if a.enfoque == "niveles" else "desviaciones de cada país"
    c1, c2 = st.columns([5, 1])
    c1.markdown(
        f'<div class="estado">Modelo vigente: <b>{ent(len(t["prep"].datos))}</b> observaciones de <b>{t["prep"].datos["iso2"].nunique()}</b> países, '
        f'{a.anio_inicio}-{a.anio_fin}, <b>{len(a.variables)}</b> variables, {atip}, {enfoque}; '
        f'<b>{t["nc"]}</b> componentes ({pct(t["acp"].acumulada[t["nc"] - 1] * 100)}), K = <b>{t["k"]}</b> (silueta {num(t["km"].silueta, 3)}).</div>',
        unsafe_allow_html=True)
    if not es_estudio():
        c2.button("Volver al estudio", on_click=restablecer, width="stretch")


def leyenda_grupos(t: dict) -> str:
    return " · ".join(f'<span style="color:{t["colores"][g]}">■</span> {n}' for g, n in sorted(t["nombres"].items()))
