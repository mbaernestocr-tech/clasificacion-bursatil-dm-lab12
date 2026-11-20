# -*- coding: utf-8 -*-
"""Carga de precios común a las misiones m1-m4 del trabajo grupal (Tema 13). Yahoo Finance con respaldo en 4 CSV (D03)."""
import numpy as np, pandas as pd, yfinance as yf
TICKERS = ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE", "XLC", "GLD", "TLT",
           "AAPL", "MSFT", "JPM", "XOM", "CAT", "KO", "PFE", "BVN", "FSM", "SCCO", "BHP", "B", "^VIX"]
SUF = "_28_activos_2010-01-04_2026-11-19.csv"
ARCH = {"cierres": "26.11.20.cierres_ajustados" + SUF, "altos": "26.11.20.altos_ajustados" + SUF,
        "bajos": "26.11.20.bajos_ajustados" + SUF, "volumenes": "26.11.20.volumenes" + SUF}


def cargar(extra=()):
    """Devuelve (cierres, altos, bajos, volumenes). Si los 4 CSV del curso están en la carpeta se usan (cifras idénticas a las de referencia);
    si no, se descargan de Yahoo Finance. Los tickers fuera de la lista de 28 (extra) se descargan siempre en vivo y se agregan."""
    import os
    if all(os.path.exists(f) for f in ARCH.values()):
        c, a, b, v = [pd.read_csv(ARCH[k], index_col=0, parse_dates=True) for k in ["cierres", "altos", "bajos", "volumenes"]]
        print("Foto de datos del curso (CSV)")
    else:
        d = yf.download(TICKERS, start="2010-01-01", end="2026-11-20", auto_adjust=True, progress=False)
        c, a, b, v = [d[k].reindex(columns=TICKERS) for k in ["Close", "High", "Low", "Volume"]]
        print("Descarga de Yahoo Finance")
    faltan = [t for t in extra if t not in c.columns]
    if faltan:
        d = yf.download(faltan, start="2010-01-01", end="2026-11-20", auto_adjust=True, progress=False)
        for tabla, k in [(c, "Close"), (a, "High"), (b, "Low"), (v, "Volume")]:
            x = d[k] if isinstance(d[k], pd.DataFrame) else d[k].to_frame(faltan[0])
            for t in faltan:
                tabla[t] = x[t].reindex(tabla.index)
    for t in (c, a, b, v):
        t.index = pd.to_datetime(t.index)
    return c, a, b, v


def variables(c, a, b, v, t, vix):
    """Variables causales de un activo (solo información al cierre del día t)."""
    cc = c[t].dropna(); aa, bb, vv = a[t].reindex(cc.index), b[t].reindex(cc.index), v[t].reindex(cc.index)
    vx = vix.reindex(cc.index).ffill(); r = np.log(cc).diff() * 100
    X = pd.DataFrame({"ret_1": r, "ret_5": r.rolling(5).sum(), "ret_22": r.rolling(22).sum(), "vol_5": r.rolling(5).std(),
                      "vol_22": r.rolling(22).std(), "vol_66": r.rolling(66).std(), "vix": vx, "d_vix": vx.diff(5),
                      "parkinson": (np.log(aa / bb) / np.sqrt(4 * np.log(2)) * 100).rolling(5).mean(),
                      "dist_sma50": (cc / cc.rolling(50).mean() - 1) * 100,
                      "rango_22": (cc - bb.rolling(22).min()) / (aa.rolling(22).max() - bb.rolling(22).min()),
                      "d_volumen": np.log(vv / vv.rolling(22).mean())}).loc["2012-01-01":]
    return cc, r, X
