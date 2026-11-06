"""Meteorología diaria por provincia (NASA POWER, reanálisis MERRA-2, sin claves).

El NO2 que ve el satélite depende tanto de lo que se emite como del tiempo:
el viento lo dispersa, la radiación lo destruye y la lluvia limpia el aire.
Estas variables sirven para quitar el efecto del tiempo antes de buscar
la señal de actividad.

Salida: data/meteo_diaria_provincias.csv.gz
"""
from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "data" / "meteo_diaria_provincias.csv.gz"
VARIABLES = ["WS10M", "WS50M", "T2M", "RH2M", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN", "PS"]
URL = "https://power.larc.nasa.gov/api/temporal/daily/point"


def main(desde: str = "20180701", hasta: str | None = None) -> None:
    hasta = hasta or (dt.date.today() - dt.timedelta(days=3)).strftime("%Y%m%d")
    prov = gpd.read_file(RAIZ / "geo" / "provincias.geojson").to_crs("EPSG:4326")
    filas = []
    for _, r in prov.iterrows():
        punto = r.geometry.representative_point()
        params = dict(
            parameters=",".join(VARIABLES), community="RE", format="JSON",
            longitude=round(punto.x, 3), latitude=round(punto.y, 3), start=desde, end=hasta,
        )
        for intento in range(5):
            try:
                j = requests.get(URL, params=params, timeout=180).json()
                break
            except Exception:
                time.sleep(10 * (intento + 1))
        datos = j["properties"]["parameter"]
        fechas = datos[VARIABLES[0]].keys()
        for f in fechas:
            fila = {"fecha": f"{f[:4]}-{f[4:6]}-{f[6:]}", "cod_prov": int(r["cod_prov"])}
            fila.update({v.lower(): datos[v][f] for v in VARIABLES})
            filas.append(fila)
        print(r["name"], len(fechas), flush=True)
    df = pd.DataFrame(filas).replace(-999.0, pd.NA)
    df.to_csv(SALIDA, index=False, float_format="%.3f")


if __name__ == "__main__":
    main()
