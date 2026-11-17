# -*- coding: utf-8 -*-
"""Compara la base con la API del Banco Mundial para Colombia.

Por defecto usa las respuestas guardadas en data/verificacion_api/ (descargadas el 16-nov-2026).
Con --descargar las vuelve a pedir a la API (es lenta: una consulta por indicador con pausas).

Uso:  python scripts/verificar_fuente.py [--descargar]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from regimenes.fuente import COLUMNAS, VARIABLES, cargar_base  # noqa: E402

CARPETA = RAIZ / "data" / "verificacion_api"
CABECERAS = {"User-Agent": "Mozilla/5.0 (compatible; taller-regimenes/1.0)"}
PROBADOS = ["inflacion", "desempleo", "pib_usd", "pib_pc", "recaudo", "tasa_real"]


def descargar(codigo: str) -> None:
    import requests
    url = f"https://api.worldbank.org/v2/country/co/indicator/{codigo}?format=json&date=2010:2023&per_page=100"
    for intento in range(5):
        try:
            r = requests.get(url, headers=CABECERAS, timeout=60)
            if r.ok and len(r.json()) > 1:
                (CARPETA / f"{codigo}.json").write_text(r.text, encoding="utf-8")
                return
        except Exception as e:  # la API corta conexiones con frecuencia
            print(f"  {codigo}: intento {intento + 1} falló ({type(e).__name__})")
        time.sleep(3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--descargar", action="store_true")
    a = ap.parse_args()
    base = cargar_base()
    co = base[base["iso2"] == "co"].set_index("anio")
    filas = []
    for v in PROBADOS:
        codigo = VARIABLES[v]["codigo"]
        if a.descargar:
            descargar(codigo)
            time.sleep(2)
        f = CARPETA / f"{codigo}.json"
        if not f.exists():
            print(f"falta {f.name}; corre con --descargar")
            continue
        api = pd.Series({int(x["date"]): x["value"] for x in json.loads(f.read_text(encoding="utf-8"))[1]}, dtype=float)
        j = pd.concat([co[v].rename("base"), api.rename("api")], axis=1).dropna()
        rel = (j["base"] - j["api"]).abs() / j["api"].abs()
        filas.append(dict(variable=v, codigo=codigo, anios_comparados=len(j), identicos=int((rel < 1e-9).sum()),
                          diferencia_maxima_pct=round(float(rel.max() * 100), 3)))
    t = pd.DataFrame(filas)
    salida = RAIZ / "resultados" / "verificacion_fuente.csv"
    salida.parent.mkdir(exist_ok=True)
    t.to_csv(salida, index=False)
    print(t.to_string(index=False))
    print("->", salida)


if __name__ == "__main__":
    main()
