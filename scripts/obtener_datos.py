# -*- coding: utf-8 -*-
"""Reconstruye la base desde la API del Banco Mundial (WDI) con las mismas columnas del archivo original.

El archivo del repositorio (data/world_bank_data_2025.csv) es el que usa el análisis. Este script sirve para
obtener los datos desde la fuente si se quiere comprobarlos o actualizarlos; el resultado se guarda aparte en
data/world_bank_data_api.csv para no reemplazar el original. Los valores recientes pueden diferir un poco
porque el Banco Mundial revisa sus series.

La API corta las ráfagas de consultas: el script pide un indicador a la vez, con pausas y reintentos, y guarda
cada respuesta en data/_cache/ para poder retomarlo.

Uso:  python scripts/obtener_datos.py                 todos los indicadores y países, 2010-2025
      python scripts/obtener_datos.py --paises co;br  solo algunos países (prueba rápida)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from regimenes.fuente import COLUMNAS, VARIABLES, cargar_paises  # noqa: E402

CACHE = RAIZ / "data" / "_cache"
CABECERAS = {"User-Agent": "Mozilla/5.0 (compatible; taller-regimenes/1.0)"}


def pedir(url: str) -> list:
    for intento in range(6):
        try:
            r = requests.get(url, headers=CABECERAS, timeout=90)
            if r.ok:
                j = r.json()
                if len(j) > 1 and j[1] is not None:
                    return j[1]
                return []
        except Exception as e:
            print(f"  intento {intento + 1}: {type(e).__name__}")
        time.sleep(4 * (intento + 1))
    raise RuntimeError(f"La API no respondió: {url}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paises", default="all", help="códigos ISO2 separados por punto y coma, o all")
    ap.add_argument("--desde", type=int, default=2010)
    ap.add_argument("--hasta", type=int, default=2025)
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    inverso = {v: k for k, v in COLUMNAS.items()}
    filas = []
    for corto, info in VARIABLES.items():
        codigo = info["codigo"]
        archivo = CACHE / f"{codigo}_{a.paises.replace(';', '-')}_{a.desde}_{a.hasta}.json"
        if archivo.exists():
            datos = json.loads(archivo.read_text(encoding="utf-8"))
        else:
            url = (f"https://api.worldbank.org/v2/country/{a.paises}/indicator/{codigo}"
                   f"?format=json&date={a.desde}:{a.hasta}&per_page=20000")
            print("pidiendo", codigo)
            datos = pedir(url)
            archivo.write_text(json.dumps(datos), encoding="utf-8")
            time.sleep(2)
        for x in datos:
            filas.append(dict(iso2=x["country"]["id"].lower(), anio=int(x["date"]), columna=inverso[corto], valor=x["value"]))
    largo = pd.DataFrame(filas)
    paises = cargar_paises()[["iso2", "pais_en"]]
    ancho = largo.pivot_table(index=["iso2", "anio"], columns="columna", values="valor", aggfunc="first").reset_index()
    ancho = ancho.merge(paises, on="iso2", how="inner")  # deja fuera los agregados regionales
    ancho = ancho.rename(columns={"pais_en": "country_name", "iso2": "country_id", "anio": "year"})
    orden = ["country_name", "country_id", "year"] + list(COLUMNAS)
    for c in orden:
        if c not in ancho:
            ancho[c] = pd.NA
    ancho = ancho[orden].sort_values(["country_name", "year"])
    salida = RAIZ / "data" / "world_bank_data_api.csv"
    ancho.to_csv(salida, index=False)
    print(f"{len(ancho)} filas, {ancho['country_id'].nunique()} países ->", salida)


if __name__ == "__main__":
    main()
