# -*- coding: utf-8 -*-
"""Gráficos de la aplicación (plotly). Cada función recibe datos ya calculados y devuelve una figura."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .fuente import etiqueta, unidad
from .interfaz import ACENTO, LINEA, PALETA, PANEL, TENUE, TINTA

FUENTE = "Public Sans, Arial, sans-serif"
TITULAR = "Archivo, Arial, sans-serif"
DIVERGENTE = [[0, "#2F6FD0"], [0.5, "#F1F2F4"], [1, "#D1335F"]]
SECUENCIAL = [[0, "#F4F7F8"], [1, ACENTO]]
GRIS_NUBE = "#C9D1D9"


def _base(fig: go.Figure, titulo: str = "", alto: int = 420, leyenda: bool = True) -> go.Figure:
    fig.update_layout(
        title=dict(text=titulo, font=dict(family=TITULAR, size=16, color=TINTA), x=0, xanchor="left", y=0.97),
        font=dict(family=FUENTE, size=13, color=TINTA), paper_bgcolor=PANEL, plot_bgcolor=PANEL, height=alto,
        margin=dict(l=60, r=24, t=58 if titulo else 24, b=52), separators=",.", showlegend=leyenda,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title=None, font=dict(size=12)),
        hoverlabel=dict(font=dict(family=FUENTE, size=12), bgcolor="white"),
    )
    fig.update_xaxes(gridcolor="#E9EDF1", zerolinecolor=LINEA, linecolor=LINEA, ticks="outside", tickcolor=LINEA)
    fig.update_yaxes(gridcolor="#E9EDF1", zerolinecolor=LINEA, linecolor=LINEA, ticks="outside", tickcolor=LINEA)
    return fig


def _c(x: float, d: int = 1) -> str:
    return f"{x:,.{d}f}".replace(",", "§").replace(".", ",").replace("§", ".")


# ---------------------------------------------------------------- datos
def cobertura(cob: pd.DataFrame) -> go.Figure:
    y = [etiqueta(v) for v in cob.index]
    txt = [[_c(v, 0) for v in fila] for fila in cob.to_numpy()]
    fig = go.Figure(go.Heatmap(z=cob.to_numpy(), x=[str(a) for a in cob.columns], y=y, zmin=0, zmax=100, colorscale=SECUENCIAL,
                               text=txt, texttemplate="%{text}", textfont=dict(size=10), xgap=2, ygap=2,
                               colorbar=dict(title="% presente", ticksuffix=" %", thickness=12),
                               hovertemplate="%{y}, %{x}: %{z:.0f} % de los países<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="top", ticks="")
    return _base(fig, "", alto=430, leyenda=False)


def distribuciones(datos: pd.DataFrame, variables: list[str], iso2: str, nombre_pais: str) -> go.Figure:
    n = len(variables)
    cols = 3
    filas = math.ceil(n / cols)
    fig = make_subplots(rows=filas, cols=cols, subplot_titles=[f"{etiqueta(v)} ({unidad(v)})" for v in variables],
                        horizontal_spacing=0.07, vertical_spacing=0.16)
    for i, v in enumerate(variables):
        r, c = i // cols + 1, i % cols + 1
        x = datos[v]
        lo, hi = x.quantile([0.005, 0.995])
        xs = x.clip(lo, hi)
        fig.add_trace(go.Histogram(x=xs, nbinsx=40, marker=dict(color="#9FB3C2", line=dict(color="white", width=1)),
                                   hovertemplate="%{x}: %{y} observaciones<extra></extra>", showlegend=False), r, c)
        med = datos.loc[datos["iso2"] == iso2, v].median()
        if not np.isnan(med):
            fig.add_vline(x=float(np.clip(med, lo, hi)), line=dict(color="#D1335F", width=2), row=r, col=c)
    fig.add_annotation(text=f"Línea roja: mediana de {nombre_pais} en sus años del panel. Ejes recortados al 0,5 % de cada cola.",
                       xref="paper", yref="paper", x=0, y=-0.13, showarrow=False, font=dict(size=11, color=TENUE), xanchor="left")
    fig.update_annotations(font=dict(family=FUENTE, size=12))
    return _base(fig, "", alto=290 * filas + 40, leyenda=False)


def correlaciones(X: pd.DataFrame) -> go.Figure:
    c = X.corr()
    et = [etiqueta(v) for v in c.columns]
    txt = [[_c(v, 2) for v in fila] for fila in c.to_numpy()]
    fig = go.Figure(go.Heatmap(z=c.to_numpy(), x=et, y=et, zmin=-1, zmax=1, colorscale=DIVERGENTE, text=txt,
                               texttemplate="%{text}", xgap=2, ygap=2, colorbar=dict(thickness=12, title="r"),
                               hovertemplate="%{y} y %{x}: %{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False)
    return _base(fig, "Correlación de Pearson entre las variables del modelo", alto=470, leyenda=False)


def cajas_escala(X: pd.DataFrame, titulo: str) -> go.Figure:
    fig = go.Figure()
    for i, v in enumerate(X.columns):
        fig.add_trace(go.Box(y=X[v], name=etiqueta(v), marker=dict(color=PALETA[i % len(PALETA)], size=3),
                             line=dict(width=1.5), boxpoints="outliers"))
    return _base(fig, titulo, alto=380, leyenda=False)


# ---------------------------------------------------------------- ACP
def sedimentacion(valores: np.ndarray) -> go.Figure:
    x = [f"CP{i + 1}" for i in range(len(valores))]
    fig = go.Figure(go.Scatter(x=x, y=valores, mode="lines+markers+text", line=dict(color=ACENTO, width=2.5),
                               marker=dict(size=9, color=ACENTO), text=[_c(v, 2) for v in valores], textposition="top center",
                               cliponaxis=False, hovertemplate="%{x}: valor propio %{y:.2f}<extra></extra>"))
    fig.add_hline(y=1, line=dict(color="#D1335F", dash="dash", width=1.5), annotation_text="Kaiser: valor propio = 1",
                  annotation_position="top right", annotation_font=dict(color="#D1335F", size=11))
    fig.update_yaxes(title="Valor propio", range=[0, float(max(valores)) * 1.18])
    fig.update_xaxes(range=[-0.4, len(valores) - 0.6])
    return _base(fig, "Gráfico de sedimentación", alto=360, leyenda=False)


def varianza(var: np.ndarray, acum: np.ndarray, nc: int) -> go.Figure:
    x = [f"CP{i + 1}" for i in range(len(var))]
    col = [ACENTO if i < nc else "#B8C4CE" for i in range(len(var))]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=var * 100, marker=dict(color=col, cornerradius=4), name="Por componente",
                         text=[_c(v * 100) + " %" for v in var], textposition="inside", insidetextanchor="end",
                         textfont=dict(color="white", size=11),
                         hovertemplate="%{x}: %{y:.1f} %<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=acum * 100, mode="lines+markers", name="Acumulada", line=dict(color=TINTA, width=2),
                             marker=dict(size=7), hovertemplate="Hasta %{x}: %{y:.1f} %<extra></extra>"))
    fig.add_hline(y=70, line=dict(color=TENUE, dash="dot", width=1), annotation_text="70 %", annotation_position="bottom right")
    fig.update_yaxes(title="% de la varianza total", range=[0, 105], ticksuffix=" %")
    return _base(fig, "Varianza explicada", alto=360)


def cargas(c: pd.DataFrame, nc: int) -> go.Figure:
    d = c.iloc[:, :nc]
    txt = [[_c(v, 2) for v in fila] for fila in d.to_numpy()]
    fig = go.Figure(go.Heatmap(z=d.to_numpy(), x=list(d.columns), y=[etiqueta(v) for v in d.index], zmin=-1, zmax=1,
                               colorscale=DIVERGENTE, text=txt, texttemplate="%{text}", xgap=3, ygap=3,
                               colorbar=dict(thickness=12, title="carga"),
                               hovertemplate="%{y} en %{x}: %{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="top")
    return _base(fig, "", alto=60 * len(d) + 90, leyenda=False)


def plano(datos: pd.DataFrame, S: np.ndarray, grupos: pd.Series, colores: dict, simbolos: dict | None, cargas_df: pd.DataFrame | None,
          resaltar: str | None, nombre_resaltado: str | None, var_pct: tuple[float, float], ejes=(0, 1), titulo: str = "") -> go.Figure:
    """Nube de observaciones en dos componentes; `grupos` es una serie de etiquetas de texto (grupo, región o periodo)."""
    i, j = ejes
    fig = go.Figure()
    d = datos.assign(x=S[:, i], y=S[:, j], g=grupos.to_numpy())
    for g in [g for g in colores if g in set(d["g"])]:
        s = d[d["g"] == g]
        fig.add_trace(go.Scattergl(
            x=s["x"], y=s["y"], mode="markers", name=str(g),
            marker=dict(size=6, color=colores[g], opacity=0.6, symbol=(simbolos or {}).get(g, "circle"), line=dict(width=0)),
            customdata=np.stack([s["pais"], s["anio"]], axis=1),
            hovertemplate="%{customdata[0]} %{customdata[1]}<extra>" + str(g) + "</extra>"))
    if resaltar:
        s = d[d["iso2"] == resaltar].sort_values("anio")
        fig.add_trace(go.Scatter(x=s["x"], y=s["y"], mode="lines+markers+text", name=nombre_resaltado,
                                 line=dict(color=TINTA, width=1.5), marker=dict(size=8, color="white", line=dict(color=TINTA, width=2)),
                                 text=[str(a)[2:] if k in (0, len(s) - 1) or a == 2020 else "" for k, a in enumerate(s["anio"])],
                                 textposition="top center", textfont=dict(size=11, color=TINTA),
                                 hovertemplate=f"{nombre_resaltado} " + "%{customdata}<extra></extra>", customdata=s["anio"]))
    if cargas_df is not None:
        rango = max(np.abs(S[:, i]).max(), np.abs(S[:, j]).max())
        esc = rango * 0.8
        for v, fila in cargas_df.iterrows():
            x1, y1 = fila.iloc[i] * esc, fila.iloc[j] * esc
            fig.add_annotation(x=x1, y=y1, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                               arrowhead=3, arrowsize=1, arrowwidth=1.6, arrowcolor="#3E4C59", text="")
            fig.add_annotation(x=x1 * 1.12, y=y1 * 1.12, text=etiqueta(v), showarrow=False, font=dict(size=11, color="#3E4C59"),
                               bgcolor="rgba(255,255,255,.75)")
    fig.update_xaxes(title=f"CP{i + 1} ({_c(var_pct[0])} %)")
    fig.update_yaxes(title=f"CP{j + 1} ({_c(var_pct[1])} %)")
    return _base(fig, titulo, alto=560)


# ---------------------------------------------------------------- K
def curvas_k(t: pd.DataFrame, k_sel: int) -> go.Figure:
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Codo: inercia intra-grupo", "Silueta promedio", "Estabilidad en submuestras (ARI)"),
                        horizontal_spacing=0.08)
    for c, (col, fmt) in enumerate((("inercia", ",.0f"), ("silueta", ".3f"), ("estabilidad", ".3f")), start=1):
        fig.add_trace(go.Scatter(x=t["k"], y=t[col], mode="lines+markers", line=dict(color=ACENTO, width=2.2), marker=dict(size=8),
                                 hovertemplate="K = %{x}: %{y:" + fmt + "}<extra></extra>", showlegend=False), 1, c)
        sel = t[t["k"] == k_sel]
        fig.add_trace(go.Scatter(x=sel["k"], y=sel[col], mode="markers", marker=dict(size=15, color="rgba(0,0,0,0)", line=dict(color="#D1335F", width=2.5)),
                                 hoverinfo="skip", showlegend=False), 1, c)
        fig.update_xaxes(title="K", dtick=1, row=1, col=c)
    fig.update_annotations(font=dict(family=TITULAR, size=13))
    return _base(fig, "", alto=330, leyenda=False)


def silueta_obs(sil: np.ndarray, etiquetas: np.ndarray, nombres: dict, colores: dict, media: float) -> go.Figure:
    fig = go.Figure()
    x0 = 0
    for g in sorted(nombres):
        v = np.sort(sil[etiquetas == g])[::-1]
        x = np.arange(x0, x0 + len(v))
        fig.add_trace(go.Bar(x=x, y=v, marker=dict(color=colores[g], line=dict(width=0)), name=nombres[g], width=1.0,
                             hovertemplate=f"{nombres[g]}<br>silueta %{{y:.3f}}<extra></extra>"))
        x0 += len(v) + 25
    fig.add_hline(y=media, line=dict(color=TINTA, dash="dash", width=1.5), annotation_text=f"promedio {_c(media, 3)}",
                  annotation_position="top right")
    fig.update_layout(bargap=0)
    fig.update_xaxes(showticklabels=False, title="Observaciones (país-año) ordenadas dentro de cada grupo", showgrid=False)
    fig.update_yaxes(title="Coeficiente de silueta", range=[-0.4, 0.8])
    return _base(fig, "", alto=360)


def perfil_z(perf: dict, nombres: dict) -> go.Figure:
    z = perf["z"]
    med = perf["medianas"]
    filas = [nombres[g] for g in z.index]
    txt = [[_c(med.loc[g, v]) for v in z.columns] for g in z.index]
    fig = go.Figure(go.Heatmap(z=z.to_numpy(), x=[etiqueta(v) for v in z.columns], y=filas, zmin=-1.5, zmax=1.5, colorscale=DIVERGENTE,
                               text=txt, texttemplate="%{text}", textfont=dict(size=13), xgap=3, ygap=3,
                               colorbar=dict(thickness=12, title="z"),
                               hovertemplate="%{y}<br>%{x}: mediana %{text} (z = %{z:.2f})<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="top")
    return _base(fig, "", alto=80 * len(filas) + 110, leyenda=False)


def reparto_anio(rep: pd.DataFrame, nombres: dict, colores: dict) -> go.Figure:
    fig = go.Figure()
    for g in sorted(nombres):
        if g not in rep:
            continue
        fig.add_trace(go.Bar(x=rep.index.astype(str), y=rep[g], name=nombres[g], marker=dict(color=colores[g], line=dict(color="white", width=1.5)),
                             hovertemplate="%{x}: %{y:.0f} %<extra>" + nombres[g] + "</extra>"))
    fig.update_layout(barmode="stack", bargap=0.18, legend=dict(traceorder="normal"))
    fig.update_yaxes(title="% de las observaciones del año", range=[0, 100], ticksuffix=" %")
    return _base(fig, "", alto=400)


def transiciones(prob: pd.DataFrame, conteo: pd.DataFrame, nombres: dict) -> go.Figure:
    et = [nombres[g] for g in prob.index]
    etc = [nombres[g] for g in prob.columns]
    txt = [[f"{_c(p, 0)} %<br>({int(n)})" for p, n in zip(fp, fn)] for fp, fn in zip(prob.to_numpy(), conteo.to_numpy())]
    fig = go.Figure(go.Heatmap(z=prob.to_numpy(), x=etc, y=et, zmin=0, zmax=100, colorscale=SECUENCIAL, text=txt, texttemplate="%{text}",
                               xgap=3, ygap=3, colorbar=dict(thickness=12, title="%", ticksuffix=" %"),
                               hovertemplate="De %{y}<br>a %{x}: %{z:.1f} %<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title="Grupo en el año t", showgrid=False)
    fig.update_xaxes(title="Grupo en el año t + 1", side="top", showgrid=False)
    return _base(fig, "", alto=90 * len(et) + 140, leyenda=False)


def cinta(datos: pd.DataFrame, etiquetas: np.ndarray, isos: list[str], nombres: dict, colores: dict, anios: list[int]) -> go.Figure:
    """Una fila por país, una celda por año, coloreada por grupo."""
    d = datos[["iso2", "pais", "anio"]].assign(g=etiquetas)
    d = d[d["iso2"].isin(isos)]
    paises = [d.loc[d["iso2"] == i, "pais"].iloc[0] for i in isos if (d["iso2"] == i).any()]
    piv = d.pivot_table(index="pais", columns="anio", values="g", aggfunc="first").reindex(index=paises, columns=anios)
    k = len(nombres)
    escala = []
    for g in range(k):
        escala += [[g / k, colores[g]], [(g + 1) / k, colores[g]]]
    txt = piv.map(lambda g: "" if pd.isna(g) else nombres[int(g)])
    fig = go.Figure(go.Heatmap(z=piv.to_numpy(dtype=float), x=[str(a) for a in anios], y=paises, zmin=-0.5, zmax=k - 0.5, colorscale=escala,
                               customdata=txt.to_numpy(), xgap=3, ygap=6, showscale=False,
                               hovertemplate="%{y} %{x}<br>%{customdata}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="top", ticks="")
    return _base(fig, "", alto=max(170, 46 * len(paises) + 90), leyenda=False)


def cajas_grupo(datos: pd.DataFrame, etiquetas: np.ndarray, var: str, nombres: dict, colores: dict) -> go.Figure:
    fig = go.Figure()
    for g in sorted(nombres):
        v = datos.loc[etiquetas == g, var]
        fig.add_trace(go.Box(y=v, name=nombres[g], marker=dict(color=colores[g], size=3), line=dict(width=1.6), boxpoints="outliers"))
    fig.update_yaxes(title=f"{etiqueta(var)} ({unidad(var)})")
    fig.update_xaxes(showticklabels=False)
    return _base(fig, "", alto=380)


def mapa(datos: pd.DataFrame, etiquetas: np.ndarray, anio: int, nombres: dict, colores: dict) -> go.Figure:
    d = datos[["iso3", "pais", "anio"]].assign(regimen=[nombres[int(g)] for g in etiquetas])
    d = d[d["anio"] == anio]
    fig = px.choropleth(d, locations="iso3", color="regimen", hover_name="pais",
                        color_discrete_map={nombres[g]: colores[g] for g in nombres},
                        category_orders={"regimen": [nombres[g] for g in sorted(nombres)]})
    fig.update_traces(marker_line_color="white", marker_line_width=0.5, hovertemplate="%{hovertext}<br>%{location}<extra></extra>")
    fig.update_geos(showcountries=True, countrycolor="#FFFFFF", showcoastlines=False, showland=True, landcolor="#E4E8EC",
                    showframe=False, projection_type="natural earth", bgcolor=PANEL)
    fig = _base(fig, "", alto=470)
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    return fig


def trayectorias(datos: pd.DataFrame, S: np.ndarray, isos: list[str], nombres_paises: dict, var_pct: tuple[float, float]) -> go.Figure:
    fig = go.Figure(go.Scattergl(x=S[:, 0], y=S[:, 1], mode="markers", marker=dict(size=5, color=GRIS_NUBE, opacity=0.5),
                                 hoverinfo="skip", name="Todas las observaciones"))
    tonos = [TINTA, "#D1335F", "#2F6FD0", "#B57C00", "#00998A", "#7B55D0"]
    for n, iso in enumerate(isos):
        m = (datos["iso2"] == iso).to_numpy()
        if not m.any():
            continue
        s = datos[m].assign(x=S[m, 0], y=S[m, 1]).sort_values("anio")
        col = tonos[n % len(tonos)]
        fig.add_trace(go.Scatter(x=s["x"], y=s["y"], mode="lines+markers+text", name=nombres_paises[iso],
                                 line=dict(color=col, width=2), marker=dict(size=7, color=col),
                                 text=[str(a)[2:] if k in (0, len(s) - 1) or a == 2020 else "" for k, a in enumerate(s["anio"])],
                                 textposition="top center", textfont=dict(size=11, color=col),
                                 customdata=s["anio"], hovertemplate=nombres_paises[iso] + " %{customdata}<extra></extra>"))
    fig.update_xaxes(title=f"CP1 ({_c(var_pct[0])} %)")
    fig.update_yaxes(title=f"CP2 ({_c(var_pct[1])} %)")
    return _base(fig, "", alto=520)
