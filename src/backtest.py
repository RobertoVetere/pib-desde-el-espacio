"""Backtest: ¿añade el NO2 visto desde el espacio información sobre la actividad
industrial que no tuviéramos ya?

Pregunta concreta, para cada comunidad autónoma y mes t:
    al final del mes t, estimar la variación anual del IPI de ese mes, que el INE
    publica unas seis semanas después.

Información disponible al final del mes t (sin mirar el futuro):
    - IPI publicado hasta t-2
    - diferencia de días laborables frente al mismo mes del año anterior
    - NO2 de Sentinel-5P del propio mes t (solo el modelo satelital)

Se entrena con 2019-07 → 2022-12 y se evalúa con 2023-01 en adelante, meses que
el modelo no ha visto. Se compara contra dos referencias:
    - ingenua: repetir el último IPI publicado (t-2)
    - base: regresión con IPI pasado + días laborables (sin satélite)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from oficiales import PROVINCIA_A_CCAA

RAIZ = Path(__file__).resolve().parents[1]
DATA = RAIZ / "data"
RES = RAIZ / "resultados"
CORTE = "2023-01"


def no2_mensual_ccaa() -> pd.DataFrame:
    d = pd.read_csv(DATA / "no2_diario_provincias.csv.gz")
    d["region"] = d["cod_prov"].map(PROVINCIA_A_CCAA)
    d = d.dropna(subset=["region"])
    d["mes"] = d["fecha"].str[:7]
    d["suma"] = d["no2_media"] * d["n_pixeles"]
    g = d.groupby(["region", "mes"]).agg(suma=("suma", "sum"), n=("n_pixeles", "sum"), dias=("fecha", "nunique"))
    g["no2"] = g["suma"] / g["n"]
    reg = g.reset_index()[["region", "mes", "no2", "n", "dias"]]
    # Total nacional (sin Ceuta y Melilla), ponderado por píxeles.
    nac = d.groupby("mes").agg(suma=("suma", "sum"), n=("n_pixeles", "sum"), dias=("fecha", "nunique")).reset_index()
    nac["no2"] = nac["suma"] / nac["n"]
    nac["region"] = "Total Nacional"
    return pd.concat([reg, nac[["region", "mes", "no2", "n", "dias"]]], ignore_index=True)


def no2_normalizado_ccaa() -> pd.DataFrame:
    """Quita al NO2 diario el efecto del tiempo, la estación y el día de la semana.

    Un modelo aprende, solo con datos anteriores a CORTE y sin el confinamiento de
    2020, cuánto NO2 cabría esperar en cada provincia dado el viento, la radiación,
    la lluvia, la temperatura, la humedad y la época del año. Lo que queda (observado
    menos esperado) es la parte que el tiempo no explica: la candidata a actividad.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    d = pd.read_csv(DATA / "no2_diario_provincias.csv.gz")
    m = pd.read_csv(DATA / "meteo_diaria_provincias.csv.gz")
    d = d.merge(m, on=["fecha", "cod_prov"], how="inner")
    f = pd.to_datetime(d["fecha"])
    d["doy_s"], d["doy_c"] = np.sin(2 * np.pi * f.dt.dayofyear / 365.25), np.cos(2 * np.pi * f.dt.dayofyear / 365.25)
    d["dia_semana"] = f.dt.weekday
    d["no2_u"] = d["no2_media"] * 1e6  # micromol/m2
    X = ["ws10m", "ws50m", "t2m", "rh2m", "prectotcorr", "allsky_sfc_sw_dwn", "ps", "doy_s", "doy_c", "dia_semana", "cod_prov"]
    confinamiento = (d["fecha"] >= "2020-03-14") & (d["fecha"] <= "2020-06-30")
    ent = d[(d["fecha"] < CORTE + "-01") & ~confinamiento]
    modelo = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, categorical_features=[X.index("cod_prov"), X.index("dia_semana")], random_state=0)
    modelo.fit(ent[X], ent["no2_u"], sample_weight=np.sqrt(ent["n_pixeles"]))
    d["esperado"] = modelo.predict(d[X])
    d["region"] = d["cod_prov"].map(PROVINCIA_A_CCAA)
    d = d.dropna(subset=["region"])
    d["mes"] = d["fecha"].str[:7]
    d["laborable"] = d["dia_semana"] < 5
    w = d["n_pixeles"]
    d["obs_w"], d["esp_w"] = d["no2_u"] * w, d["esperado"] * w

    def agregar(g: pd.DataFrame) -> pd.Series:
        lab, fin = g[g["laborable"]], g[~g["laborable"]]
        anom = 100 * (g["obs_w"].sum() / g["esp_w"].sum() - 1)
        a_lab = 100 * (lab["obs_w"].sum() / lab["esp_w"].sum() - 1) if len(lab) else np.nan
        a_fin = 100 * (fin["obs_w"].sum() / fin["esp_w"].sum() - 1) if len(fin) else np.nan
        return pd.Series({"no2_anomalia": anom, "no2_contraste_semana": a_lab - a_fin})

    reg = d.groupby(["region", "mes"]).apply(agregar).reset_index()
    nac = d.groupby("mes").apply(agregar).reset_index()
    nac["region"] = "Total Nacional"
    return pd.concat([reg, nac], ignore_index=True)


def _festivos_nacionales(anio: int) -> set[pd.Timestamp]:
    fijos = ["01-01", "01-06", "05-01", "08-15", "10-12", "11-01", "12-06", "12-08", "12-25"]
    f = {pd.Timestamp(f"{anio}-{m}") for m in fijos}
    # Viernes Santo (y Jueves Santo, festivo en casi todas las comunidades).
    a, b, c = anio % 19, anio // 100, anio % 100
    d, e = b // 4, b % 4
    g = (8 * b + 13) // 25
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 19 * l) // 433
    mes = (h + l - 7 * m + 90) // 25
    dia = (h + l - 7 * m + 33 * mes + 19) % 32
    pascua = pd.Timestamp(anio, mes, dia)
    f |= {pascua - pd.Timedelta(days=2), pascua - pd.Timedelta(days=3)}
    return f


def laborables(mes: str) -> int:
    ini = pd.Timestamp(mes + "-01")
    dias = pd.date_range(ini, ini + pd.offsets.MonthEnd(0))
    fest = _festivos_nacionales(ini.year)
    return int(sum(1 for x in dias if x.weekday() < 5 and x not in fest))


def construir_panel() -> pd.DataFrame:
    ipi = pd.read_csv(DATA / "ipi_ccaa.csv")
    no2 = no2_mensual_ccaa()
    p = ipi.merge(no2, on=["region", "mes"], how="left")
    p = p.merge(no2_normalizado_ccaa(), on=["region", "mes"], how="left").sort_values(["region", "mes"])
    p["periodo"] = pd.PeriodIndex(p["mes"], freq="M")
    p = p.set_index(["region", "periodo"]).sort_index()

    def desplazar(col: str, k: int) -> pd.Series:
        return p.groupby(level=0)[col].shift(k)

    p["ipi_t2"] = desplazar("ipi_var_anual", 2)
    p["ipi_t3"] = desplazar("ipi_var_anual", 3)
    p["ipi_t14"] = desplazar("ipi_var_anual", 14)  # mismo mes del año anterior, ya publicado
    p["no2_hace_12"] = desplazar("no2", 12)
    p["no2_var_anual"] = 100 * np.log(p["no2"] / p["no2_hace_12"])
    p["no2_var_anual_t1"] = desplazar("no2_var_anual", 1)
    p["anom_var_anual"] = p["no2_anomalia"] - desplazar("no2_anomalia", 12)
    p["anom_var_anual_t1"] = desplazar("anom_var_anual", 1)
    p["contraste_var_anual"] = p["no2_contraste_semana"] - desplazar("no2_contraste_semana", 12)
    lab = {m: laborables(m) for m in p.reset_index()["mes"].unique()}
    p["dif_laborables"] = [lab[m] - lab.get(f"{int(m[:4]) - 1}{m[4:]}", lab[m]) for m in p["mes"]]
    return p.reset_index()


VARIANTES = {
    "base (sin satélite)": [],
    "NO2 en bruto": ["no2_var_anual", "no2_var_anual_t1"],
    "NO2 sin efecto del tiempo": ["anom_var_anual", "anom_var_anual_t1"],
    "NO2 sin tiempo + contraste laborable/fin de semana": ["anom_var_anual", "anom_var_anual_t1", "contraste_var_anual"],
}
BASE = ["ipi_t2", "ipi_t3", "dif_laborables"]
COVID = ("2020-03", "2021-06")


def evaluar(p: pd.DataFrame) -> dict:
    todas = sorted({c for v in VARIANTES.values() for c in v})
    p = p.dropna(subset=["ipi_var_anual", *BASE, *todas]).copy()
    p = p[p["region"] != "Total Nacional"]
    prueba = p[p["mes"] >= CORTE].copy()
    prueba["pred_ingenua"] = prueba["ipi_t2"]
    alphas = np.logspace(-2, 4, 30)
    rng = np.random.default_rng(0)
    meses = prueba["mes"].unique()
    remuestreos = [rng.choice(meses, len(meses)) for _ in range(2000)]

    def err_mes(col: str) -> pd.Series:
        return (prueba["ipi_var_anual"] - prueba[col]).abs().groupby(prueba["mes"]).sum()

    tabla = []
    for sin_covid in (False, True):
        ent = p[p["mes"] < CORTE]
        if sin_covid:
            ent = ent[~ent["mes"].between(*COVID)]
        sufijo = " | entrenado sin COVID" if sin_covid else ""
        cols_pred = {}
        for nombre, extra in VARIANTES.items():
            X = BASE + extra
            m = RidgeCV(alphas=alphas).fit(ent[X], ent["ipi_var_anual"])
            col = f"pred::{nombre}{sufijo}"
            prueba[col] = m.predict(prueba[X])
            cols_pred[nombre] = col
        eb = err_mes(cols_pred["base (sin satélite)"])
        for nombre, col in cols_pred.items():
            es = err_mes(col)
            mejoras = [1 - es.loc[mm].sum() / eb.loc[mm].sum() for mm in remuestreos]
            lo, hi = np.percentile(mejoras, [2.5, 97.5])
            tabla.append({
                "modelo": nombre + sufijo,
                "mae": float(np.mean(np.abs(prueba["ipi_var_anual"] - prueba[col]))),
                "mejora_vs_base_pct": 100 * (1 - es.sum() / eb.sum()),
                "ic95_lo": 100 * lo,
                "ic95_hi": 100 * hi,
                "observaciones_entreno": len(ent),
            })
    tabla = pd.DataFrame(tabla)
    RES.mkdir(exist_ok=True)
    tabla.to_csv(RES / "comparativa_modelos.csv", index=False, float_format="%.2f")
    cols = ["region", "mes", "ipi_var_anual", "pred_ingenua", *[c for c in prueba.columns if c.startswith("pred::")]]
    prueba[cols].to_csv(RES / "predicciones_prueba.csv", index=False, float_format="%.3f")
    resumen = {
        "prueba": f"{prueba['mes'].min()} → {prueba['mes'].max()} ({len(prueba)} observaciones, {len(meses)} meses)",
        "mae_ingenua": float(np.mean(np.abs(prueba["ipi_var_anual"] - prueba["pred_ingenua"]))),
        "modelos": tabla.to_dict(orient="records"),
    }
    (RES / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    return resumen


def correlacion_pib(p: pd.DataFrame) -> pd.DataFrame:
    """Serie nacional trimestral: NO2 frente a PIB (solo descriptivo)."""
    nac = p[p["region"] == "Total Nacional"].copy()
    nac["trimestre"] = nac["periodo"].dt.year.astype(str) + "-T" + nac["periodo"].dt.quarter.astype(str)
    no2_t = nac.groupby("trimestre")["no2"].mean()
    t = pd.DataFrame({"no2": no2_t})
    t["no2_var_anual"] = 100 * np.log(t["no2"] / t["no2"].shift(4))
    pib = pd.read_csv(DATA / "pib_trimestral.csv").set_index("trimestre")
    t = t.join(pib, how="inner").dropna()
    t.to_csv(RES / "pib_vs_no2_trimestral.csv", float_format="%.3f")
    return t


def main() -> None:
    p = construir_panel()
    r = evaluar(p)
    t = correlacion_pib(p)
    print(json.dumps(r, indent=2, ensure_ascii=False))
    print("Correlación trimestral PIB vs NO2 (var. anual):", round(t["pib_var_anual"].corr(t["no2_var_anual"]), 3))
    print("Sin 2020-2021:", round(t.loc[~t.index.str[:4].isin(["2020", "2021"])].pipe(lambda x: x["pib_var_anual"].corr(x["no2_var_anual"])), 3))


if __name__ == "__main__":
    main()
