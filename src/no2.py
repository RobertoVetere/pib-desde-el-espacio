"""Extrae NO2 troposférico diario de Sentinel-5P (TROPOMI) por provincia española.

Fuente: COGs públicos de MEEO en AWS (s3://meeo-s5p, producto OFFL L2__NO2___),
sin claves ni cuentas. Solo se descarga la ventana que cubre España gracias a las
peticiones HTTP por rangos de los Cloud-Optimized GeoTIFF.

Salida: data/no2_diario_provincias.csv.gz con columnas
    fecha, cod_prov, no2_media (mol/m2), n_pixeles
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.features import rasterize
from rasterio.windows import from_bounds

BUCKET = "https://meeo-s5p.s3.amazonaws.com"
PREFIX = "COGT/OFFL/L2__NO2___"
# Península, Baleares y Canarias.
BBOX = (-18.5, 27.5, 4.5, 44.0)
QA_MIN = 75  # recomendación ESA: qa_value >= 0.75
RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "data" / "no2_diario_provincias.csv.gz"

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MULTIRANGE", "YES")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
os.environ.setdefault("VSI_CACHE", "TRUE")

_PROVINCIAS = gpd.read_file(RAIZ / "geo" / "provincias.geojson").to_crs("EPSG:4326")
_PROVINCIAS["id"] = _PROVINCIAS["cod_prov"].astype(int)


def orbitas_del_dia(dia: dt.date) -> list[tuple[str, str]]:
    """Devuelve (columna_troposferica, qa) de las órbitas diurnas que pasan sobre España."""
    url = f"{BUCKET}/?list-type=2&prefix={PREFIX}/{dia:%Y/%m/%d}/"
    claves = re.findall(r"<Key>([^<]*_4326\.tif)</Key>", requests.get(url, timeout=60).text)
    pares = []
    for inicio in sorted({c.split("____")[1][:15] for c in claves}):
        hora = int(inicio[9:11])
        if not 10 <= hora <= 14:  # el paso diurno sobre España es hacia las 13 h UTC
            continue
        trop = [c for c in claves if inicio in c and "PRODUCT_nitrogendioxide_tropospheric_column" in c]
        qa = [c for c in claves if inicio in c and "PRODUCT_qa_value" in c]
        if trop and qa:
            pares.append((trop[0], qa[0]))
    return pares


def _leer(clave: str):
    with rasterio.open(f"/vsicurl/{BUCKET}/{clave}") as src:
        ventana = from_bounds(*BBOX, src.transform)
        datos = src.read(1, window=ventana, boundless=True, fill_value=src.nodata)
        return datos, src.window_transform(ventana), src.nodata


def procesar_dia(dia: dt.date) -> pd.DataFrame:
    sumas: dict[int, float] = {}
    cuentas: dict[int, int] = {}
    for trop, qa in orbitas_del_dia(dia):
        no2, transformada, nodata = _leer(trop)
        calidad, _, _ = _leer(qa)
        valido = (no2 != nodata) & np.isfinite(no2) & (calidad != 255) & (calidad >= QA_MIN)
        if not valido.any():
            continue
        ids = rasterize(
            ((g, i) for g, i in zip(_PROVINCIAS.geometry, _PROVINCIAS["id"])),
            out_shape=no2.shape,
            transform=transformada,
            fill=0,
            all_touched=False,
            dtype="int32",
        )
        m = valido & (ids > 0)
        if not m.any():
            continue
        s = np.bincount(ids[m], weights=no2[m].astype("float64"))
        c = np.bincount(ids[m])
        for i in np.nonzero(c)[0]:
            sumas[i] = sumas.get(i, 0.0) + s[i]
            cuentas[i] = cuentas.get(i, 0) + int(c[i])
    filas = [
        {"fecha": dia.isoformat(), "cod_prov": i, "no2_media": sumas[i] / cuentas[i], "n_pixeles": cuentas[i]}
        for i in sorted(cuentas)
    ]
    return pd.DataFrame(filas, columns=["fecha", "cod_prov", "no2_media", "n_pixeles"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--desde", default="2018-07-01")
    p.add_argument("--hasta", default=(dt.date.today() - dt.timedelta(days=5)).isoformat())
    p.add_argument("--hilos", type=int, default=16)
    a = p.parse_args()

    previo = pd.read_csv(SALIDA, dtype={"fecha": str}) if SALIDA.exists() else pd.DataFrame()
    hechos = set(previo["fecha"]) if len(previo) else set()
    d0, d1 = dt.date.fromisoformat(a.desde), dt.date.fromisoformat(a.hasta)
    dias = [d0 + dt.timedelta(n) for n in range((d1 - d0).days + 1)]
    dias = [d for d in dias if d.isoformat() not in hechos]
    print(f"{len(dias)} días por procesar", flush=True)

    nuevos, fallos = [], []
    with ThreadPoolExecutor(a.hilos) as ex:
        futuros = {ex.submit(procesar_dia, d): d for d in dias}
        for n, f in enumerate(as_completed(futuros), 1):
            d = futuros[f]
            try:
                nuevos.append(f.result())
            except Exception as e:  # una órbita corrupta no debe parar la serie
                fallos.append(d)
                print(f"fallo {d}: {e}", file=sys.stderr, flush=True)
            if n % 100 == 0 or n == len(dias):
                print(f"{n}/{len(dias)}", flush=True)
                _guardar(previo, nuevos)
    _guardar(previo, nuevos)
    print(f"Hecho. Fallos: {len(fallos)}")


def _guardar(previo: pd.DataFrame, nuevos: list[pd.DataFrame]) -> None:
    if not nuevos:
        return
    df = pd.concat([previo, *nuevos], ignore_index=True)
    df = df.drop_duplicates(["fecha", "cod_prov"]).sort_values(["fecha", "cod_prov"])
    SALIDA.parent.mkdir(exist_ok=True)
    df.to_csv(SALIDA, index=False, float_format="%.4e")


if __name__ == "__main__":
    main()
