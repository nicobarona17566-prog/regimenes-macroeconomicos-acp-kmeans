# -*- coding: utf-8 -*-
"""De la base al panel que entra al modelo.

Cada observación es un país en un año. Las decisiones viven en `Ajustes`, para que la aplicación y los
scripts usen exactamente las mismas reglas: ventana de años, variables, tratamiento de valores anómalos,
transformación de la inflación, enfoque (niveles o desviaciones respecto del promedio de cada país) y
estandarización.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .fuente import BASE

ID = ["iso2", "iso3", "pais", "region", "ingreso", "anio"]
TOPE_FISCAL = 80.0      # ingreso o gasto del gobierno por encima de este % del PIB
TOPE_EXTERNO = 60.0     # saldo de cuenta corriente por encima de este % del PIB, en valor absoluto
MIN_ANIOS_DESVIACION = 5


@dataclass(frozen=True)
class Ajustes:
    anio_inicio: int = 2010
    anio_fin: int = 2023
    variables: tuple[str, ...] = tuple(BASE)
    atipicos: str = "excluir"          # "excluir", "winsorizar" o "ninguno"
    log_inflacion: bool = True
    enfoque: str = "niveles"            # "niveles" o "desviaciones"


ESTUDIO = Ajustes()


def log_con_signo(x: pd.Series) -> pd.Series:
    """ln(1 + |x|) con el signo de x: comprime las hiperinflaciones y conserva las deflaciones."""
    return np.sign(x) * np.log1p(np.abs(x))


def regla_anomalos(df: pd.DataFrame) -> pd.Series:
    """Marca filas con cifras fiscales o externas imposibles para una economía típica."""
    m = pd.Series(False, index=df.index)
    for v in ("ingreso_publico", "gasto_publico"):
        if v in df:
            m |= df[v] > TOPE_FISCAL
    if "cuenta_corriente" in df:
        m |= df["cuenta_corriente"].abs() > TOPE_EXTERNO
    return m


@dataclass
class Preparacion:
    ajustes: Ajustes
    datos: pd.DataFrame                  # filas que entran al modelo, con identificadores y valores originales
    X: pd.DataFrame                      # variables transformadas, antes de estandarizar
    Z: np.ndarray                        # matriz estandarizada (media 0, desviación 1)
    escalador: StandardScaler
    etapas: list[tuple[str, int, int]]   # (paso, observaciones, países)
    sin_datos: pd.DataFrame              # observaciones de la ventana que no tienen todas las variables
    anomalas: pd.DataFrame               # observaciones excluidas por la regla de anómalos
    recortes: dict = field(default_factory=dict)  # límites de la winsorización, si se usó

    @property
    def variables(self) -> list[str]:
        return list(self.ajustes.variables)


def preparar(panel: pd.DataFrame, aj: Ajustes = ESTUDIO) -> Preparacion:
    vars_ = list(aj.variables)
    etapas = [("Base completa", len(panel), panel["iso2"].nunique())]

    ventana = panel[(panel["anio"] >= aj.anio_inicio) & (panel["anio"] <= aj.anio_fin)]
    etapas.append((f"Años {aj.anio_inicio}-{aj.anio_fin}", len(ventana), ventana["iso2"].nunique()))

    completas = ventana.dropna(subset=vars_)
    sin_datos = ventana[ventana[vars_].isna().any(axis=1)]
    etapas.append(("Con las variables completas", len(completas), completas["iso2"].nunique()))

    anomalas = completas.iloc[0:0]
    if aj.atipicos == "excluir":
        marca = regla_anomalos(completas)
        anomalas = completas[marca]
        completas = completas[~marca]
        etapas.append(("Sin valores anómalos", len(completas), completas["iso2"].nunique()))

    if aj.enfoque == "desviaciones":
        n = completas.groupby("iso2")["anio"].transform("size")
        completas = completas[n >= MIN_ANIOS_DESVIACION]
        etapas.append((f"Países con {MIN_ANIOS_DESVIACION} años o más", len(completas), completas["iso2"].nunique()))

    extra = [c for c in ("pib_pc",) if c in completas and c not in vars_]
    datos = completas[ID + vars_ + extra].reset_index(drop=True)
    X = datos[vars_].astype(float).copy()

    if aj.log_inflacion:
        for v in ("inflacion", "deflactor"):
            if v in X:
                X[v] = log_con_signo(X[v])
    if aj.log_inflacion and "tasa_real" in X:
        X["tasa_real"] = log_con_signo(X["tasa_real"])

    recortes = {}
    if aj.atipicos == "winsorizar":
        for v in vars_:
            lo, hi = X[v].quantile([0.01, 0.99])
            recortes[v] = (float(lo), float(hi))
            X[v] = X[v].clip(lo, hi)

    if aj.enfoque == "desviaciones":
        X = X - X.groupby(datos["iso2"]).transform("mean")

    esc = StandardScaler().fit(X)
    Z = esc.transform(X)
    return Preparacion(aj, datos, X, Z, esc, etapas, sin_datos[ID + vars_].reset_index(drop=True),
                       anomalas[ID + vars_].reset_index(drop=True), recortes)


def faltantes_por_variable(panel: pd.DataFrame, aj: Ajustes, candidatas: list[str]) -> pd.DataFrame:
    """Cuántas observaciones y países quedan al agregar cada variable candidata a las del modelo."""
    ventana = panel[(panel["anio"] >= aj.anio_inicio) & (panel["anio"] <= aj.anio_fin)]
    base = list(aj.variables)
    filas = []
    for v in candidatas:
        cols = base if v in base else base + [v]
        c = ventana.dropna(subset=cols)
        filas.append(dict(variable=v, en_modelo=v in base, observaciones=len(c), paises=c["iso2"].nunique(),
                          presentes=float(ventana[v].notna().mean() * 100)))
    return pd.DataFrame(filas)
