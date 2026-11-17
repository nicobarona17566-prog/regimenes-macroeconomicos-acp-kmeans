# -*- coding: utf-8 -*-
"""Lectura de la base y catálogo de variables.

La base es un extracto de los Indicadores del Desarrollo Mundial (WDI) del Banco Mundial con una fila por
país y año (217 economías, 2010-2025) y 13 indicadores macroeconómicos. Este módulo la carga, renombra las
columnas, le agrega el nombre en español, la región y el nivel de ingreso, y calcula su cobertura.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "data"
ARCHIVO = DATOS / "world_bank_data_2025.csv"
PAISES = DATOS / "paises.csv"

# columna del archivo -> nombre corto
COLUMNAS = {
    "Inflation (CPI %)": "inflacion",
    "GDP (Current USD)": "pib_usd",
    "GDP per Capita (Current USD)": "pib_pc",
    "Unemployment Rate (%)": "desempleo",
    "Interest Rate (Real, %)": "tasa_real",
    "Inflation (GDP Deflator, %)": "deflactor",
    "GDP Growth (% Annual)": "crecimiento",
    "Current Account Balance (% GDP)": "cuenta_corriente",
    "Government Expense (% of GDP)": "gasto_publico",
    "Government Revenue (% of GDP)": "ingreso_publico",
    "Tax Revenue (% of GDP)": "recaudo",
    "Gross National Income (USD)": "ingreso_nacional",
    "Public Debt (% of GDP)": "deuda_publica",
}

# catálogo: etiqueta corta, descripción, código WDI, unidad y papel en el estudio
VARIABLES = {
    "inflacion": dict(etiqueta="Inflación", larga="Inflación al consumidor, variación anual", codigo="FP.CPI.TOTL.ZG", unidad="%", papel="modelo"),
    "crecimiento": dict(etiqueta="Crecimiento", larga="Crecimiento del PIB real, variación anual", codigo="NY.GDP.MKTP.KD.ZG", unidad="%", papel="modelo"),
    "desempleo": dict(etiqueta="Desempleo", larga="Desempleo, % de la fuerza laboral (estimación modelada de la OIT)", codigo="SL.UEM.TOTL.ZS", unidad="%", papel="modelo"),
    "cuenta_corriente": dict(etiqueta="Cuenta corriente", larga="Saldo de la cuenta corriente, % del PIB", codigo="BN.CAB.XOKA.GD.ZS", unidad="% PIB", papel="modelo"),
    "ingreso_publico": dict(etiqueta="Ingreso público", larga="Ingresos del gobierno central sin donaciones, % del PIB", codigo="GC.REV.XGRT.GD.ZS", unidad="% PIB", papel="modelo"),
    "gasto_publico": dict(etiqueta="Gasto público", larga="Gasto del gobierno central, % del PIB", codigo="GC.XPN.TOTL.GD.ZS", unidad="% PIB", papel="modelo"),
    "tasa_real": dict(etiqueta="Tasa real", larga="Tasa de interés real de los préstamos bancarios", codigo="FR.INR.RINR", unidad="%", papel="opcional"),
    "recaudo": dict(etiqueta="Recaudo", larga="Ingresos tributarios, % del PIB", codigo="GC.TAX.TOTL.GD.ZS", unidad="% PIB", papel="opcional"),
    "deflactor": dict(etiqueta="Deflactor", larga="Inflación medida por el deflactor del PIB", codigo="NY.GDP.DEFL.KD.ZG", unidad="%", papel="opcional"),
    "deuda_publica": dict(etiqueta="Deuda pública", larga="Deuda del gobierno central, % del PIB", codigo="GC.DOD.TOTL.GD.ZS", unidad="% PIB", papel="descartada"),
    "pib_pc": dict(etiqueta="PIB per cápita", larga="PIB per cápita en dólares corrientes", codigo="NY.GDP.PCAP.CD", unidad="USD", papel="descriptiva"),
    "pib_usd": dict(etiqueta="PIB", larga="PIB en dólares corrientes", codigo="NY.GDP.MKTP.CD", unidad="USD", papel="descartada"),
    "ingreso_nacional": dict(etiqueta="Ingreso nacional", larga="Ingreso nacional bruto en dólares corrientes", codigo="NY.GNP.MKTP.CD", unidad="USD", papel="descartada"),
}
INDICADORES = list(COLUMNAS.values())
BASE = ["inflacion", "crecimiento", "desempleo", "cuenta_corriente", "ingreso_publico", "gasto_publico"]
OPCIONALES = ["tasa_real", "recaudo", "deflactor"]

NOMBRES_CORTOS = {
    "cd": "R. D. del Congo", "cg": "Congo", "eg": "Egipto", "fm": "Micronesia", "hk": "Hong Kong", "ir": "Irán",
    "kr": "Corea del Sur", "la": "Laos", "mo": "Macao", "mf": "San Martín (Francia)", "pr": "Puerto Rico",
    "kp": "Corea del Norte", "ps": "Cisjordania y Gaza", "so": "Somalia", "sx": "Sint Maarten", "cf": "Rep. Centroafricana",
    "vg": "Islas Vírgenes Británicas", "vi": "Islas Vírgenes (EE. UU.)", "ye": "Yemen", "vc": "San Vicente y Granadinas",
}
INGRESO_CORTO = {"Países de ingreso mediano bajo": "Ingreso mediano bajo", "Países de ingreso bajo": "Ingreso bajo"}
REGION_CORTA = {"Oriente Medio, Norte de África, Afganistán y Pakistán": "Oriente Medio y Norte de África",
                "África al sur del Sahara": "África subsahariana"}
ORDEN_INGRESO = ["Ingreso bajo", "Ingreso mediano bajo", "Ingreso mediano alto", "Ingreso alto"]


def etiqueta(v: str) -> str:
    return VARIABLES[v]["etiqueta"]


def unidad(v: str) -> str:
    return VARIABLES[v]["unidad"]


def cargar_base(ruta: Path | str = ARCHIVO) -> pd.DataFrame:
    """La base tal como viene, con nombres de columna cortos."""
    df = pd.read_csv(ruta)
    df = df.rename(columns=COLUMNAS).rename(columns={"country_name": "pais_en", "country_id": "iso2", "year": "anio"})
    df["iso2"] = df["iso2"].str.lower().str.strip()
    df["pais_en"] = df["pais_en"].str.strip()
    return df


def cargar_paises(ruta: Path | str = PAISES) -> pd.DataFrame:
    p = pd.read_csv(ruta, keep_default_na=False)
    for c in p.columns:
        p[c] = p[c].astype(str).str.strip()
    return p


def panel() -> pd.DataFrame:
    """Base con nombre en español, región y nivel de ingreso."""
    df = cargar_base()
    meta = cargar_paises()[["iso2", "iso3", "pais", "region", "ingreso"]]
    out = df.merge(meta, on="iso2", how="left")
    out["pais"] = out["pais"].fillna(out["pais_en"])
    cols = ["iso2", "iso3", "pais", "pais_en", "region", "ingreso", "anio"] + INDICADORES
    return out[cols].sort_values(["pais", "anio"]).reset_index(drop=True)


def cobertura(df: pd.DataFrame, variables: list[str] | None = None) -> pd.DataFrame:
    """Porcentaje de datos presentes por variable (filas) y año (columnas)."""
    variables = variables or INDICADORES
    return (df.groupby("anio")[variables].apply(lambda g: g.notna().mean() * 100)).T


def resumen_calidad(df: pd.DataFrame) -> dict:
    return dict(
        filas=len(df), paises=df["iso2"].nunique(), anio_min=int(df["anio"].min()), anio_max=int(df["anio"].max()),
        indicadores=len(INDICADORES), duplicados=int(df.duplicated(["iso2", "anio"]).sum()),
        celdas_vacias=float(df[INDICADORES].isna().mean().mean() * 100),
    )
