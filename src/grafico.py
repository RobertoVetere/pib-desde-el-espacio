"""Gráfico para el README: IPI nacional frente a NO2 nacional (variación anual)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from backtest import construir_panel

RAIZ = Path(__file__).resolve().parents[1]
AZUL, NARANJA = "#2a78d6", "#eb6834"
FONDO, TEXTO, TEXTO_2, REJILLA = "#fcfcfb", "#1f1f1e", "#6b6a64", "#e6e5e0"


def main() -> None:
    p = construir_panel()
    n = p[(p["region"] == "Total Nacional") & (p["mes"] >= "2019-07")].dropna(subset=["no2_var_anual"])
    x = pd.to_datetime(n["mes"] + "-01")

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150, facecolor=FONDO)
    ax.set_facecolor(FONDO)
    ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-30"), color="#efeee9", lw=0)
    ax.text(pd.Timestamp("2020-03-15"), n["ipi_var_anual"].max() - 2, "COVID", color=TEXTO_2, fontsize=9)
    ax.axhline(0, color="#b9b8b0", lw=1)
    ax.plot(x, n["ipi_var_anual"], color=AZUL, lw=2, label="Producción industrial (INE)")
    ax.plot(x, n["no2_var_anual"], color=NARANJA, lw=2, label="NO₂ desde el espacio (Sentinel-5P)")
    ax.text(x.iloc[-1] + pd.Timedelta(days=20), n["ipi_var_anual"].iloc[-1], "IPI", color=TEXTO, fontsize=9, va="center")
    ax.text(x.iloc[-1] + pd.Timedelta(days=20), n["no2_var_anual"].iloc[-1], "NO₂", color=TEXTO, fontsize=9, va="center")
    ax.set_ylabel("Variación anual (%)", color=TEXTO_2)
    ax.grid(axis="y", color=REJILLA, lw=0.8)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#b9b8b0")
    ax.tick_params(colors=TEXTO_2, length=0)
    lo = min(n["ipi_var_anual"].min(), n["no2_var_anual"].min()); hi = max(n["ipi_var_anual"].max(), n["no2_var_anual"].max())
    ax.set_ylim(lo - 5, hi + 5)
    ax.legend(loc="lower right", frameon=False, labelcolor=TEXTO, fontsize=9)
    ax.set_title(
        "El NO₂ ve el confinamiento, pero fuera de los grandes choques no sigue a la industria",
        loc="left", color=TEXTO, fontsize=11.5, pad=12,
    )
    fig.text(0.01, 0.01, "España, total nacional, mensual. Fuentes: INE (IPI, tabla 70177); ESA/Copernicus Sentinel-5P vía MEEO.",
             color=TEXTO_2, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    (RAIZ / "resultados").mkdir(exist_ok=True)
    fig.savefig(RAIZ / "resultados" / "ipi_vs_no2_nacional.png", facecolor=FONDO)


if __name__ == "__main__":
    main()
