# -*- coding: utf-8 -*-
"""Genera las tablas, métricas y figuras del README con la configuración del estudio, sin abrir la aplicación.

Usa exactamente las mismas funciones que la aplicación (paquete `regimenes`).

Uso:  python scripts/generar_resultados.py
Salida: resultados/*.csv, resultados/metricas.json y resultados/figuras/*.png
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from regimenes import fuente as F  # noqa: E402
from regimenes import modelo as M  # noqa: E402
from regimenes import preparacion as P  # noqa: E402
from regimenes.robustez import tabla_robustez  # noqa: E402

N_COMP, K = 3, 3
OUT = RAIZ / "resultados"
FIG = OUT / "figuras"
TINTA, TENUE, ACENTO = "#13202E", "#5B6875", "#0B5F66"
COLOR = {M.NOMBRES["pequeno"]: "#B57C00", M.NOMBRES["grande"]: "#2F6FD0", M.NOMBRES["fragil"]: "#D1335F", M.NOMBRES["recesion"]: "#7B55D0"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#C9D1D9", "axes.labelcolor": TINTA, "xtick.color": TENUE, "ytick.color": TENUE,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 11,
                     "axes.titlelocation": "left", "figure.dpi": 150, "axes.formatter.use_locale": False})


def coma(x, d=1):
    return f"{x:,.{d}f}".replace(",", "§").replace(".", ",").replace("§", ".")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    panel = F.panel()
    prep = P.preparar(panel, P.ESTUDIO)
    d = prep.datos
    r = M.acp(prep.Z, prep.variables)
    S = r.coordenadas(N_COMP)
    tk = M.evaluar_k(S)
    km = M.segmentar(S, K)
    perf = M.perfiles(d, km.etiquetas, prep.variables)
    nom = M.nombres_grupos(perf, True)
    rep = M.reparto_por_anio(d, km.etiquetas)
    conteo, prob = M.transiciones(d, km.etiquetas)
    co = M.trayectoria(d, km.etiquetas, S, "co")
    rob = tabla_robustez(panel, P.ESTUDIO, N_COMP, K)

    pd.DataFrame(prep.etapas, columns=["paso", "observaciones", "paises"]).to_csv(OUT / "etapas.csv", index=False)
    prep.anomalas.to_csv(OUT / "observaciones_anomalas.csv", index=False)
    pd.DataFrame({"componente": [f"CP{i + 1}" for i in range(len(r.varianza))], "valor_propio": r.valores_propios,
                  "varianza_pct": r.varianza * 100, "acumulada_pct": r.acumulada * 100}).to_csv(OUT / "varianza.csv", index=False)
    r.cargas.to_csv(OUT / "cargas.csv")
    tk.to_csv(OUT / "eleccion_k.csv", index=False)
    perf["medianas"].rename(index=nom).assign(observaciones=perf["tam"].to_numpy(), paises=perf["paises"].to_numpy(),
                                              pib_pc_mediano=perf["pib_pc"].to_numpy()).to_csv(OUT / "perfiles_grupos.csv")
    rep.rename(columns=nom).to_csv(OUT / "reparto_por_anio.csv")
    prob.rename(index=nom, columns=nom).to_csv(OUT / "transiciones_pct.csv")
    d[["iso3", "pais", "region", "ingreso", "anio"]].assign(grupo=[nom[g] for g in km.etiquetas], silueta=km.silueta_obs,
                                                            **{f"CP{i + 1}": S[:, i] for i in range(N_COMP)}).to_csv(OUT / "asignaciones.csv", index=False)
    co.assign(grupo=[nom[g] for g in co["grupo"]]).to_csv(OUT / "trayectoria_colombia.csv", index=False)
    M.gemelos_periodo(d, S, "co", n=10).to_csv(OUT / "parecidos_colombia.csv", index=False)
    rob.to_csv(OUT / "robustez.csv", index=False)

    cambios = pd.Series({i: M.cambios(M.trayectoria(d, km.etiquetas, S, i)) for i in d["iso2"].unique()})
    metricas = dict(
        observaciones=len(d), paises=int(d["iso2"].nunique()), anios=[int(d["anio"].min()), int(d["anio"].max())],
        variables=prep.variables, componentes=N_COMP, varianza_retenida_pct=round(float(r.acumulada[N_COMP - 1] * 100), 2),
        valores_propios=[round(float(v), 3) for v in r.valores_propios], k=K, silueta=round(km.silueta, 4),
        estabilidad_k=dict(zip(tk["k"].astype(int).astype(str), tk["estabilidad"].round(3))),
        silueta_k=dict(zip(tk["k"].astype(int).astype(str), tk["silueta"].round(3))),
        grupos={nom[g]: int(n) for g, n in perf["tam"].items()},
        permanencia_pct=round(float(np.trace(conteo.to_numpy()) / conteo.to_numpy().sum() * 100), 2),
        paises_sin_cambios_pct=round(float((cambios == 0).mean() * 100), 2),
        silueta_negativa_pct=round(float((km.silueta_obs < 0).mean() * 100), 2),
        colombia=dict(grupo=nom[int(co["grupo"].mode().iloc[0])], anios=len(co), cambios=M.cambios(co),
                      parecidos=M.gemelos_periodo(d, S, "co")["pais"].tolist()),
    )
    (OUT / "metricas.json").write_text(json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8")

    # figura 1: varianza
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    x = np.arange(1, len(r.varianza) + 1)
    ax[0].plot(x, r.valores_propios, "o-", color=ACENTO, lw=2)
    ax[0].axhline(1, color="#D1335F", ls="--", lw=1)
    ax[0].text(len(x), 1.05, "Kaiser", color="#D1335F", ha="right", fontsize=9)
    ax[0].set(title="Valores propios", xticks=x, xticklabels=[f"CP{i}" for i in x], ylabel="Valor propio")
    ax[1].bar(x, r.varianza * 100, color=[ACENTO if i <= N_COMP else "#B8C4CE" for i in x])
    ax[1].plot(x, r.acumulada * 100, "o-", color=TINTA, lw=1.6)
    ax[1].axhline(70, color=TENUE, ls=":", lw=1)
    for i, v in zip(x, r.varianza * 100):
        ax[1].text(i, v + 1.5, coma(v), ha="center", fontsize=8, color=TINTA)
    ax[1].set(title=f"Varianza explicada: {coma(r.acumulada[N_COMP - 1] * 100)} % con {N_COMP} componentes", xticks=x,
              xticklabels=[f"CP{i}" for i in x], ylabel="% de la varianza", ylim=(0, 105))
    fig.tight_layout()
    fig.savefig(FIG / "varianza.png")
    plt.close(fig)

    # figura 2: elección de K
    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    for a_, col, tit in zip(ax, ["inercia", "silueta", "estabilidad"], ["Codo: inercia", "Silueta promedio", "Estabilidad (ARI en submuestras)"]):
        a_.plot(tk["k"], tk[col], "o-", color=ACENTO, lw=2)
        s = tk[tk["k"] == K]
        a_.scatter(s["k"], s[col], s=160, facecolors="none", edgecolors="#D1335F", lw=2, zorder=3)
        a_.set(title=tit, xlabel="K", xticks=tk["k"])
    fig.tight_layout()
    fig.savefig(FIG / "eleccion_k.png")
    plt.close(fig)

    # figura 3: reparto por año
    fig, ax = plt.subplots(figsize=(10, 3.4))
    abajo = np.zeros(len(rep))
    for g in sorted(nom):
        ax.bar(rep.index.astype(str), rep[g], bottom=abajo, color=COLOR.get(nom[g]), label=nom[g], edgecolor="white", lw=1)
        abajo += rep[g].to_numpy()
    ax.set(ylabel="% de las observaciones del año", ylim=(0, 100), title="Peso de cada grupo, año por año")
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "reparto_por_anio.png")
    plt.close(fig)

    # figura 4: cinta de países
    isos = ["co", "br", "cl", "pe", "mx", "us", "de", "es"]
    anios = list(range(2010, 2024))
    fig, ax = plt.subplots(figsize=(10, 3.3))
    for fila, iso in enumerate(isos):
        t = M.trayectoria(d, km.etiquetas, S, iso)
        for _, rr in t.iterrows():
            ax.add_patch(plt.Rectangle((rr["anio"] - 0.45, fila - 0.38), 0.9, 0.76, color=COLOR.get(nom[int(rr["grupo"])])))
    ax.set(xlim=(2009.4, 2023.6), ylim=(len(isos) - 0.5, -0.5), xticks=anios, yticks=range(len(isos)),
           yticklabels=[d.loc[d["iso2"] == i, "pais"].iloc[0] for i in isos], title="Grupo de cada país, año por año")
    ax.tick_params(axis="x", labelsize=8)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_visible(False)
    manijas = [plt.Rectangle((0, 0), 1, 1, color=COLOR.get(nom[g])) for g in sorted(nom)]
    ax.legend(manijas, [nom[g] for g in sorted(nom)], ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "cinta_paises.png")
    plt.close(fig)

    # figura 5: plano de los grupos
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for g in sorted(nom):
        m = km.etiquetas == g
        ax.scatter(S[m, 0], S[m, 1], s=8, alpha=0.55, color=COLOR.get(nom[g]), label=nom[g], lw=0)
    ax.plot(co["CP1"], co["CP2"], "-o", color=TINTA, ms=4, lw=1.2, label="Colombia 2010-2023")
    ax.set(xlabel=f"CP1, tamaño del Estado ({coma(r.varianza[0] * 100)} %)", ylabel=f"CP2, holgura externa y laboral ({coma(r.varianza[1] * 100)} %)",
           title="Observaciones país-año en los dos primeros componentes")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "plano_grupos.png")
    plt.close(fig)

    print(json.dumps(metricas, ensure_ascii=False, indent=1)[:1500])
    print("->", OUT)


if __name__ == "__main__":
    main()
