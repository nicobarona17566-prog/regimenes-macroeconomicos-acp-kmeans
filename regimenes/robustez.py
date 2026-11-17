# -*- coding: utf-8 -*-
"""Prueba de robustez: cuánto cambian la partición y el lugar de Colombia al variar una decisión a la vez."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from . import fuente as F
from . import modelo as M
from . import preparacion as P


def tabla_robustez(panel: pd.DataFrame, a: P.Ajustes, nc: int, k: int) -> pd.DataFrame:
    """Cambia una decisión a la vez respecto de `a` y compara la partición resultante con la de `a`."""
    base_prep = P.preparar(panel, a)
    base = M.segmentar(M.acp(base_prep.Z, base_prep.variables).coordenadas(nc), k)
    ref = base_prep.datos[["iso2", "anio"]].assign(g0=base.etiquetas)
    alternativas = [("Configuración vigente", a, nc, k)]
    alternativas += [(f"{n} componentes", a, n, k) for n in (2, 3, 4) if n != nc and n <= len(a.variables)]
    alternativas += [(f"K = {kk}", a, nc, kk) for kk in (k - 1, k + 1) if kk >= 2]
    if a.atipicos != "winsorizar":
        alternativas.append(("Winsorizar en lugar de excluir", replace(a, atipicos="winsorizar"), nc, k))
    alternativas.append(("Sin logaritmo en la inflación" if a.log_inflacion else "Con logaritmo en la inflación",
                         replace(a, log_inflacion=not a.log_inflacion), nc, k))
    for extra in ("tasa_real", "recaudo"):
        if extra not in a.variables:
            alternativas.append((f"Agregando {F.etiqueta(extra).lower()}", replace(a, variables=tuple(a.variables) + (extra,)), nc, k))
    if a.enfoque == "niveles":
        alternativas.append(("Desviaciones del promedio de cada país", replace(a, enfoque="desviaciones"), nc, k))
    filas = []
    for nombre, aj, n, kk in alternativas:
        pr = P.preparar(panel, aj)
        ra = M.acp(pr.Z, pr.variables)
        km = M.segmentar(ra.coordenadas(min(n, len(pr.variables))), kk)
        dd = pr.datos[["iso2", "anio"]].assign(g=km.etiquetas)
        j = dd.merge(ref, on=["iso2", "anio"])
        tr = dd[dd["iso2"] == "co"].sort_values("anio")
        g = tr["g"].to_numpy()
        juntos = {}
        for otro in ("br", "mx", "pe"):
            o = dd[dd["iso2"] == otro][["anio", "g"]].merge(tr[["anio", "g"]], on="anio", suffixes=("_o", "_c"))
            juntos[otro] = float((o["g_o"] == o["g_c"]).mean() * 100) if len(o) else np.nan
        cp = dd.groupby("iso2")["g"].apply(lambda s: int((s.to_numpy()[1:] != s.to_numpy()[:-1]).sum()))
        cnt, _ = M.transiciones(pr.datos, km.etiquetas)
        perm = float(np.trace(cnt.to_numpy()) / cnt.to_numpy().sum() * 100)
        filas.append({"Alternativa": nombre, "Observaciones": len(pr.datos), "Silueta": km.silueta,
                      "ARI": M.comparar_particiones(j["g0"], j["g"]),
                      "Permanencia (%)": perm, "Sin cambios (%)": float((cp == 0).mean() * 100),
                      "Cambios COL": float((g[1:] != g[:-1]).sum()) if len(g) else np.nan,
                      "COL-BRA (%)": juntos["br"], "COL-MEX (%)": juntos["mx"],
                      "COL-PER (%)": juntos["pe"]})
    return pd.DataFrame(filas)
