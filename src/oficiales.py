"""Descarga las series oficiales contra las que se mide el acierto.

- IPI (Índice de Producción Industrial) del INE: variación anual, mensual,
  nacional y por comunidad autónoma. Tabla 70177.
- PIB trimestral del INE: variación anual del índice de volumen, corregido de
  estacionalidad y calendario. Serie CNTR6654.

API pública del INE (servicios.ine.es/wstempus), sin claves.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import requests

RAIZ = Path(__file__).resolve().parents[1]
DATA = RAIZ / "data"
INE = "https://servicios.ine.es/wstempus/js/ES"

# Código INE de provincia -> nombre de comunidad tal y como aparece en el IPI.
PROVINCIA_A_CCAA = {
    4: "Andalucía", 11: "Andalucía", 14: "Andalucía", 18: "Andalucía", 21: "Andalucía",
    23: "Andalucía", 29: "Andalucía", 41: "Andalucía",
    22: "Aragón", 44: "Aragón", 50: "Aragón",
    33: "Asturias, Principado de",
    7: "Balears, Illes",
    35: "Canarias", 38: "Canarias",
    39: "Cantabria",
    5: "Castilla y León", 9: "Castilla y León", 24: "Castilla y León", 34: "Castilla y León",
    37: "Castilla y León", 40: "Castilla y León", 42: "Castilla y León", 47: "Castilla y León",
    49: "Castilla y León",
    2: "Castilla - La Mancha", 13: "Castilla - La Mancha", 16: "Castilla - La Mancha",
    19: "Castilla - La Mancha", 45: "Castilla - La Mancha",
    8: "Cataluña", 17: "Cataluña", 25: "Cataluña", 43: "Cataluña",
    3: "Comunitat Valenciana", 12: "Comunitat Valenciana", 46: "Comunitat Valenciana",
    6: "Extremadura", 10: "Extremadura",
    15: "Galicia", 27: "Galicia", 32: "Galicia", 36: "Galicia",
    28: "Madrid, Comunidad de",
    30: "Murcia, Región de",
    31: "Navarra, Comunidad Foral de",
    1: "País Vasco", 20: "País Vasco", 48: "País Vasco",
    26: "Rioja, La",
    # Ceuta (51) y Melilla (52) no tienen IPI propio.
}

def ipi_por_ccaa() -> pd.DataFrame:
    hoy = dt.date.today()
    url = f"{INE}/DATOS_TABLA/70177?date=20170101:{hoy:%Y%m%d}"
    series = requests.get(url, timeout=120).json()
    filas = []
    for s in series:
        nombre = s["Nombre"]
        if "Total industria. Variación anual" not in nombre:
            continue
        region = nombre.split(".")[0].strip()
        for d in s["Data"]:
            if d.get("Valor") is None:
                continue
            filas.append({"region": region, "mes": f"{d['Anyo']}-{d['FK_Periodo']:02d}", "ipi_var_anual": d["Valor"]})
    df = pd.DataFrame(filas).drop_duplicates(["region", "mes"]).sort_values(["region", "mes"])
    return df

def pib_trimestral() -> pd.DataFrame:
    url = f"{INE}/DATOS_SERIE/CNTR6654?nult=200"
    datos = requests.get(url, timeout=120).json()["Data"]
    filas = []
    for d in datos:
        # En el INE, FK_Periodo 19..22 son los trimestres T1..T4.
        trimestre = d["FK_Periodo"] - 18
        filas.append({"trimestre": f"{d['Anyo']}-T{trimestre}", "pib_var_anual": d["Valor"]})
    return pd.DataFrame(filas).drop_duplicates("trimestre").sort_values("trimestre")

def main() -> None:
    DATA.mkdir(exist_ok=True)
    ipi = ipi_por_ccaa()
    ipi.to_csv(DATA / "ipi_ccaa.csv", index=False)
    print(f"IPI: {len(ipi)} filas, {ipi['region'].nunique()} regiones, {ipi['mes'].min()} → {ipi['mes'].max()}")
    pib = pib_trimestral()
    pib.to_csv(DATA / "pib_trimestral.csv", index=False)
    print(f"PIB: {len(pib)} trimestres, {pib['trimestre'].min()} → {pib['trimestre'].max()}")

if __name__ == "__main__":
    main()
