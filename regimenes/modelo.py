# -*- coding: utf-8 -*-
"""ACP, elección de K, K-Means y lectura de los regímenes en el tiempo."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score

from .fuente import etiqueta

SEMILLA = 42
LETRAS = "ABCDEFGHIJ"


# ---------------------------------------------------------------- ACP
@dataclass
class ResultadoACP:
    pca: PCA
    valores_propios: np.ndarray
    varianza: np.ndarray        # proporción explicada por componente
    acumulada: np.ndarray
    cargas: pd.DataFrame        # correlación variable-componente (vector propio por raíz del valor propio)
    vectores: pd.DataFrame      # vectores propios orientados
    scores: np.ndarray          # coordenadas de todas las observaciones en todos los componentes

    def coordenadas(self, n: int) -> np.ndarray:
        return self.scores[:, :n]


def acp(Z: np.ndarray, variables: list[str]) -> ResultadoACP:
    pca = PCA().fit(Z)
    vec = pca.components_.copy()
    # orientación: la variable con la carga más grande en valor absoluto queda con signo positivo
    for i in range(vec.shape[0]):
        if vec[i, np.argmax(np.abs(vec[i]))] < 0:
            vec[i] *= -1
    lam = pca.explained_variance_ratio_ * len(variables)   # valores propios de la matriz de correlaciones
    nombres = [f"CP{i + 1}" for i in range(vec.shape[0])]
    cargas = pd.DataFrame(vec.T * np.sqrt(lam), index=variables, columns=nombres)
    scores = (Z - Z.mean(axis=0)) @ vec.T
    return ResultadoACP(pca, lam, pca.explained_variance_ratio_, np.cumsum(pca.explained_variance_ratio_), cargas,
                        pd.DataFrame(vec.T, index=variables, columns=nombres), scores)


def sugerir_componentes(r: ResultadoACP, umbral: float = 0.70) -> dict:
    kaiser = int((r.valores_propios > 1).sum())
    acum = int(np.argmax(r.acumulada >= umbral) + 1)
    # codo: mayor caída relativa entre valores propios consecutivos, sin contar el último
    caidas = r.valores_propios[:-2] - r.valores_propios[1:-1]
    codo = int(np.argmax(caidas) + 1) if len(caidas) else 1
    return dict(kaiser=max(kaiser, 1), acumulada=acum, codo=codo)


# ---------------------------------------------------------------- elección de K
def evaluar_k(S: np.ndarray, kmin: int = 2, kmax: int = 8, repeticiones: int = 20, fraccion: float = 0.8) -> pd.DataFrame:
    """Inercia, silueta y estabilidad (índice de Rand ajustado entre la partición completa y particiones
    hechas sobre submuestras del 80 %) para cada K."""
    rng = np.random.default_rng(SEMILLA)
    n = len(S)
    filas = []
    for k in range(kmin, kmax + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA).fit(S)
        aris = []
        for b in range(repeticiones):
            idx = rng.choice(n, size=int(n * fraccion), replace=False)
            sub = KMeans(n_clusters=k, n_init=5, random_state=b).fit_predict(S[idx])
            aris.append(adjusted_rand_score(km.labels_[idx], sub))
        filas.append(dict(k=k, inercia=float(km.inertia_), silueta=float(silhouette_score(S, km.labels_)),
                          estabilidad=float(np.mean(aris)), estabilidad_min=float(np.min(aris))))
    t = pd.DataFrame(filas)
    t["caida_inercia"] = -t["inercia"].pct_change() * 100
    return t


# ---------------------------------------------------------------- K-Means
@dataclass
class ResultadoKMeans:
    k: int
    etiquetas: np.ndarray        # 0..k-1, ordenadas por tamaño (0 = el grupo más grande)
    centroides: np.ndarray
    silueta: float
    silueta_obs: np.ndarray
    inercia: float


def segmentar(S: np.ndarray, k: int) -> ResultadoKMeans:
    km = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA).fit(S)
    orden = np.argsort(-np.bincount(km.labels_, minlength=k), kind="stable")
    mapa = {viejo: nuevo for nuevo, viejo in enumerate(orden)}
    et = np.array([mapa[e] for e in km.labels_])
    cent = km.cluster_centers_[orden]
    return ResultadoKMeans(k, et, cent, float(silhouette_score(S, et)), silhouette_samples(S, et), float(km.inertia_))


def perfiles(datos: pd.DataFrame, etiquetas: np.ndarray, variables: list[str]) -> dict:
    d = datos.assign(grupo=etiquetas)
    med = d.groupby("grupo")[variables].median()
    general = d[variables].median()
    # z de las medianas de cada grupo frente a la distribución completa (unidades de desviación estándar)
    z = (med - d[variables].mean()) / d[variables].std(ddof=0)
    return dict(medianas=med, medias=d.groupby("grupo")[variables].mean(), z=z, general=general,
                tam=d["grupo"].value_counts().sort_index(), paises=d.groupby("grupo")["iso2"].nunique(),
                pib_pc=d.groupby("grupo")["pib_pc"].median() if "pib_pc" in d else None)


def rasgos(z: pd.Series, n: int = 2) -> str:
    """Los rasgos más marcados de un grupo, en palabras."""
    partes = []
    for v in z.abs().sort_values(ascending=False).index[:n]:
        nivel = "alto" if z[v] > 0 else "bajo"
        if v in ("cuenta_corriente",):
            nivel = "positiva" if z[v] > 0 else "negativa"
        if v in ("inflacion", "crecimiento"):
            nivel = "alta" if z[v] > 0 else "baja"
        partes.append(f"{etiqueta(v).lower()} {nivel}")
    return " · ".join(partes)


# Nombres de los grupos. Con las variables base y el enfoque en niveles, los grupos de K = 3 y K = 4 se
# nombran por reglas sobre sus medianas, así que el nombre no depende del número que KMeans le asigne.
NOMBRES = {
    "pequeno": "Estado pequeño, crecimiento alto",
    "grande": "Estado grande, superávit externo",
    "fragil": "Desempleo alto, déficit externo",
    "recesion": "Recesión con gasto alto",
}
CLAVES = {v: k for k, v in NOMBRES.items()}


def nombres_grupos(perf: dict, nombrables: bool) -> dict[int, str]:
    med = perf["medianas"]
    k = len(med)
    cols = {"crecimiento", "ingreso_publico", "cuenta_corriente"}
    if nombrables and k in (3, 4) and cols <= set(med.columns):
        rest = list(med.index)
        out = {}
        if k == 4:
            rec = med.loc[rest, "crecimiento"].idxmin(); rest.remove(rec); out[rec] = NOMBRES["recesion"]
        peq = med.loc[rest, "ingreso_publico"].idxmin(); rest.remove(peq); out[peq] = NOMBRES["pequeno"]
        gra = med.loc[rest, "cuenta_corriente"].idxmax(); rest.remove(gra); out[gra] = NOMBRES["grande"]
        out[rest[0]] = NOMBRES["fragil"]
        return out
    return {g: f"Grupo {LETRAS[g]}" for g in med.index}


# ---------------------------------------------------------------- el tiempo
def reparto_por_anio(datos: pd.DataFrame, etiquetas: np.ndarray) -> pd.DataFrame:
    """Porcentaje de las observaciones de cada año que cae en cada grupo."""
    return pd.crosstab(datos["anio"], etiquetas, normalize="index") * 100


def transiciones(datos: pd.DataFrame, etiquetas: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Matriz de paso de un año al siguiente para el mismo país: conteos y probabilidades por fila."""
    d = datos[["iso2", "anio"]].assign(g=etiquetas)
    sig = d.assign(anio=d["anio"] - 1).rename(columns={"g": "g_sig"})
    par = d.merge(sig, on=["iso2", "anio"])
    conteo = pd.crosstab(par["g"], par["g_sig"])
    prob = conteo.div(conteo.sum(axis=1), axis=0) * 100
    return conteo, prob


def trayectoria(datos: pd.DataFrame, etiquetas: np.ndarray, S: np.ndarray, iso2: str) -> pd.DataFrame:
    m = (datos["iso2"] == iso2).to_numpy()
    t = datos[m].copy()
    t["grupo"] = etiquetas[m]
    for i in range(min(3, S.shape[1])):
        t[f"CP{i + 1}"] = S[m, i]
    return t.sort_values("anio").reset_index(drop=True)


def cambios(trayecto: pd.DataFrame) -> int:
    g = trayecto["grupo"].to_numpy()
    return int((g[1:] != g[:-1]).sum())


def gemelos_anio(datos: pd.DataFrame, S: np.ndarray, iso2: str, anio: int, n: int = 5) -> pd.DataFrame:
    """Los países más parecidos a `iso2` en el mismo año, por distancia en el espacio de componentes."""
    m = ((datos["iso2"] == iso2) & (datos["anio"] == anio)).to_numpy()
    if not m.any():
        return pd.DataFrame()
    otros = ((datos["anio"] == anio) & (datos["iso2"] != iso2)).to_numpy()
    dist = np.linalg.norm(S[otros] - S[m][0], axis=1)
    t = datos[otros][["iso2", "pais", "region"]].assign(distancia=dist)
    return t.nsmallest(n, "distancia").reset_index(drop=True)


def gemelos_periodo(datos: pd.DataFrame, S: np.ndarray, iso2: str, n: int = 5, min_anios: int = 5) -> pd.DataFrame:
    """Países cuyo punto promedio en el espacio de componentes queda más cerca del de `iso2`."""
    cols = [f"c{i}" for i in range(S.shape[1])]
    d = pd.concat([datos[["iso2", "pais", "region"]], pd.DataFrame(S, columns=cols)], axis=1)
    g = d.groupby(["iso2", "pais", "region"])[cols].agg(["mean", "size"])
    medias = g.xs("mean", axis=1, level=1)
    tam = g.xs("size", axis=1, level=1).iloc[:, 0]
    medias = medias[tam >= min_anios]
    if iso2 not in medias.index.get_level_values(0):
        return pd.DataFrame()
    base = medias.xs(iso2, level=0).to_numpy()[0]
    otros = medias[medias.index.get_level_values(0) != iso2]
    dist = np.linalg.norm(otros.to_numpy() - base, axis=1)
    t = otros.reset_index()[["iso2", "pais", "region"]].assign(distancia=dist, anios=tam[otros.index].to_numpy())
    return t.nsmallest(n, "distancia").reset_index(drop=True)


def acompanantes(datos: pd.DataFrame, etiquetas: np.ndarray, iso2: str) -> pd.Series:
    """Porcentaje de los años del país en que cada otro país cae en su mismo grupo."""
    d = datos[["iso2", "pais", "anio"]].assign(g=etiquetas)
    base = d[d["iso2"] == iso2][["anio", "g"]]
    j = d[d["iso2"] != iso2].merge(base, on="anio", suffixes=("", "_base"))
    j["igual"] = j["g"] == j["g_base"]
    comunes = j.groupby("pais")["igual"].agg(["mean", "size"])
    comunes = comunes[comunes["size"] >= max(3, len(base) // 2)]
    return (comunes["mean"] * 100).sort_values(ascending=False)


def comparar_particiones(base: np.ndarray, otra: np.ndarray) -> float:
    return float(adjusted_rand_score(base, otra))
