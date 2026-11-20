# =========================================================
# Misión 4 · Integración: sistema de recomendaciones de bolsa para el portafolio del grupo (clasificación walk-forward)
# Fase CRISP-DM: Modelado · Evaluación · Despliegue | Datos: REALES (Yahoo Finance) | Semilla global: 2026
# Uso: python m4_sistema_recomendaciones_portafolio.py BVN,FSM,SCCO,BHP,B,AAPL,JPM,XOM,KO,PFE   (por defecto, el portafolio de referencia)
# Escribe dashboard_clasificacion.json, recomendaciones_5_sesiones.csv y m4_figura_N.png
# =========================================================
import sys
import matplotlib
matplotlib.use("Agg")
# Librerías del laboratorio
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
)

warnings.filterwarnings("ignore")
SEMILLA = 2026  # semilla global: reproducibilidad
rng = np.random.default_rng(SEMILLA)
plt.rcParams.update(
    {"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3, "image.cmap": "viridis"}
)

# Librerías de esta guía
import json
from xgboost import XGBClassifier

# Precios ajustados de 28 activos hasta el cierre del 19-nov-2026. Si los 4 CSV del curso (carpeta D03) están junto al notebook se usan
# (las cifras coinciden con las de la guía); si no, se descargan de Yahoo Finance (pueden variar en centésimas si Yahoo revisó precios)
import os

TICKERS = [
    "SPY",
    "QQQ",
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLY",
    "XLP",
    "XLI",
    "XLU",
    "XLB",
    "XLRE",
    "XLC",
    "GLD",
    "TLT",
    "AAPL",
    "MSFT",
    "JPM",
    "XOM",
    "CAT",
    "KO",
    "PFE",
    "BVN",
    "FSM",
    "SCCO",
    "BHP",
    "B",
    "^VIX",
]
ARCH = {
    "cierres": "26.11.20.cierres_ajustados_28_activos_2010-01-04_2026-11-19.csv",
    "altos": "26.11.20.altos_ajustados_28_activos_2010-01-04_2026-11-19.csv",
    "bajos": "26.11.20.bajos_ajustados_28_activos_2010-01-04_2026-11-19.csv",
    "volumenes": "26.11.20.volumenes_28_activos_2010-01-04_2026-11-19.csv",
}
if all(os.path.exists(f) for f in ARCH.values()):
    cierres, altos, bajos, volumenes = [
        pd.read_csv(ARCH[k], index_col=0, parse_dates=True)
        for k in ["cierres", "altos", "bajos", "volumenes"]
    ]
    print("Se usó la foto de datos del curso (CSV)")
else:
    try:
        d = yf.download(
            TICKERS,
            start="2010-01-01",
            end="2026-11-20",
            auto_adjust=True,
            progress=False,
        )
        cierres, altos, bajos, volumenes = [
            d[c].reindex(columns=TICKERS) for c in ["Close", "High", "Low", "Volume"]
        ]
        print("Se descargó de Yahoo Finance")
    except Exception as e:
        raise SystemExit(
            "Falló la descarga de Yahoo Finance (%s): suba los 4 CSV del curso junto "
            "al notebook"
            % e
        )
for t in (cierres, altos, bajos, volumenes):
    t.index = pd.to_datetime(t.index)
print(
    "Dimensiones:",
    cierres.shape,
    "| desde",
    cierres.index.min().date(),
    "hasta",
    cierres.index.max().date(),
)
print(
    "Último cierre de SPY:",
    round(float(cierres["SPY"].dropna().iloc[-1]), 2),
    "| VIX:",
    round(float(cierres["^VIX"].dropna().iloc[-1]), 2),
)

# Nombres de los activos
NOMBRES = {
    "SPY": "S&P 500 (ETF)",
    "QQQ": "Nasdaq-100 (ETF)",
    "XLK": "Tecnología",
    "XLF": "Finanzas",
    "XLE": "Energía",
    "XLV": "Salud",
    "XLY": "Consumo discrecional",
    "XLP": "Consumo básico",
    "XLI": "Industriales",
    "XLU": "Servicios públicos",
    "XLB": "Materiales",
    "XLRE": "Inmobiliario",
    "XLC": "Comunicaciones",
    "GLD": "Oro",
    "TLT": "Bonos del Tesoro 20+",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "JPM": "JPMorgan",
    "XOM": "ExxonMobil",
    "CAT": "Caterpillar",
    "KO": "Coca-Cola",
    "PFE": "Pfizer",
    "BVN": "Buenaventura (Perú)",
    "FSM": "Fortuna Mining",
    "SCCO": "Southern Copper",
    "BHP": "BHP",
    "B": "Barrick Mining",
    "^VIX": "Índice VIX",
}

# Panel (fecha, activo) con variables causales y un objetivo relativo a la mediana del portafolio
PORTAFOLIO = sys.argv[1].split(",") if len(sys.argv) > 1 else ["BVN", "FSM", "SCCO", "BHP", "B", "AAPL", "JPM", "XOM", "KO", "PFE"]
vix = cierres["^VIX"].ffill()
bloques = {}
for t in PORTAFOLIO:
    c = cierres[t].dropna()
    a, b = altos[t].reindex(c.index), bajos[t].reindex(c.index)
    rr = np.log(c).diff() * 100
    bloques[t] = pd.DataFrame(
        {
            "ret_1": rr,
            "ret_5": rr.rolling(5).sum(),
            "ret_22": rr.rolling(22).sum(),
            "vol_22": rr.rolling(22).std(),
            "dist_sma50": (c / c.rolling(50).mean() - 1) * 100,
            "rango_22": (c - b.rolling(22).min())
            / (a.rolling(22).max() - b.rolling(22).min()),
            "vix": vix.reindex(c.index),
            "d_vix": vix.reindex(c.index).diff(5),
            "fwd5": c.shift(-5) / c - 1,
        }
    )
panel_x = (
    pd.concat(bloques, names=["activo", "fecha"])
    .swaplevel()
    .sort_index()
    .loc["2012-01-01":]
)
panel_x = panel_x.dropna(
    subset=[
        "ret_1",
        "ret_5",
        "ret_22",
        "vol_22",
        "dist_sma50",
        "rango_22",
        "vix",
        "d_vix",
    ]
)
cuenta = panel_x.groupby(level="fecha").size()
panel_x = panel_x[
    panel_x.index.get_level_values("fecha").isin(
        cuenta[cuenta == len(PORTAFOLIO)].index
    )
]  # solo fechas con los 10 activos
g = panel_x.groupby(level="fecha")
panel_x["rel_ret_22"] = panel_x["ret_22"] - g["ret_22"].transform("median")
panel_x["rel_vol_22"] = panel_x["vol_22"] / g["vol_22"].transform("median")
panel_x["rel_dist"] = panel_x["dist_sma50"] - g["dist_sma50"].transform("median")
VARS = [
    "ret_1",
    "ret_5",
    "ret_22",
    "vol_22",
    "dist_sma50",
    "rango_22",
    "vix",
    "d_vix",
    "rel_ret_22",
    "rel_vol_22",
    "rel_dist",
]
panel = panel_x.dropna(subset=["fwd5"]).copy()
panel["y"] = (
    panel["fwd5"] > panel.groupby(level="fecha")["fwd5"].transform("median")
).astype(int)
fechas = panel.index.get_level_values("fecha").unique().sort_values()
print(
    "Panel:",
    panel.shape,
    "| fechas:",
    len(fechas),
    "| de",
    fechas.min().date(),
    "a",
    fechas.max().date(),
    "| tasa base de y:",
    round(panel["y"].mean(), 3),
)
print(
    "Última fecha con variables (para la recomendación):",
    panel_x.index.get_level_values("fecha").max().date(),
)

# Walk-forward desde mayo de 2022
i0 = int(np.searchsorted(fechas, pd.Timestamp("2022-05-09")))
reb = list(range(i0, len(fechas), 5))
modelos = {
    "Logística": lambda: make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, C=0.1)
    ),
    "XGBoost": lambda: XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=2,
        subsample=0.7,
        min_child_weight=50,
        random_state=SEMILLA,
        n_jobs=1,
    ),
}
probs = []  # una fila por (fecha de rebalanceo, activo)
for k, pos in enumerate(reb):
    if k % 13 == 0:
        entr = panel.loc[fechas[: pos - 5]]
        ajustados = {n: f().fit(entr[VARS], entr["y"]) for n, f in modelos.items()}
    dia = panel.loc[fechas[pos]]
    fila = pd.DataFrame(
        {n: m.predict_proba(dia[VARS])[:, 1] for n, m in ajustados.items()},
        index=dia.index,
    )
    fila["Conjunto"] = fila.mean(axis=1)
    fila["fecha"] = fechas[pos]
    fila["fwd5"] = dia["fwd5"].values
    fila["y"] = dia["y"].values
    probs.append(fila)
probs = pd.concat(probs)
probs.index.name = "activo"
probs = probs.reset_index().set_index(["fecha", "activo"])
print(
    "Rebalanceos:",
    len(reb),
    "| de",
    fechas[reb[0]].date(),
    "a",
    fechas[reb[-1]].date(),
    "| reentrenamientos:",
    len(range(0, len(reb), 13)),
)
filas = [
    [
        n,
        roc_auc_score(probs["y"], probs[n]),
        accuracy_score(probs["y"], (probs[n] >= 0.5).astype(int)),
    ]
    for n in ["Logística", "XGBoost", "Conjunto"]
]
print(
    pd.DataFrame(
        filas, columns=["modelo", "AUC de ranking (walk-forward)", "exactitud"]
    )
    .round(3)
    .to_string(index=False),
    "| tasa base:",
    round(probs["y"].mean(), 3),
)

# Retornos semanales de la estrategia top-3 con costos, y de las referencias
COSTO = 0.001  # 10 puntos básicos por unidad negociada


def retornos_top3(columna):
    ret, pesos_prev = [], pd.Series(0.0, index=PORTAFOLIO)
    for f, d in probs.groupby(level="fecha"):
        d = d.droplevel("fecha")
        pesos = pd.Series(0.0, index=PORTAFOLIO)
        pesos[d[columna].nlargest(3).index] = 1 / 3
        ret.append(
            (
                f,
                float((pesos * d["fwd5"]).sum())
                - COSTO * float((pesos - pesos_prev).abs().sum()),
            )
        )
        pesos_prev = pesos
    return pd.Series(dict(ret))


semanal = pd.DataFrame(
    {n: retornos_top3(n) for n in ["Logística", "XGBoost", "Conjunto"]}
)
semanal["1/N (10 activos)"] = probs.groupby(level="fecha")["fwd5"].mean()
spy = cierres["SPY"].dropna()
semanal["SPY"] = (spy.shift(-5) / spy - 1).reindex(semanal.index)
azar = np.empty((1000, len(semanal)))
fw = probs["fwd5"].unstack("activo")[PORTAFOLIO].values
for s in range(1000):
    elegidos = np.array(
        [rng.choice(len(PORTAFOLIO), 3, replace=False) for _ in range(len(fw))]
    )
    azar[s] = (
        np.take_along_axis(fw, elegidos, axis=1).mean(axis=1)
        - COSTO * 2 * (3 - 0.9) / 3
    )  # rotación esperada: 0,9 de los 3 activos se repiten por azar


def metricas(r):
    r = r.dropna()
    eq = (1 + r).cumprod()
    return pd.Series(
        {
            "retorno anual %": ((eq.iloc[-1]) ** (52 / len(r)) - 1) * 100,
            "volatilidad anual %": r.std() * np.sqrt(52) * 100,
            "Sharpe": r.mean() / r.std() * np.sqrt(52),
            "caída máxima %": ((eq / eq.cummax()) - 1).min() * 100,
            "semanas con ganancia %": (r > 0).mean() * 100,
        }
    )


tabla_bt = pd.DataFrame({n: metricas(semanal[n]) for n in semanal.columns}).T
sh_azar = azar.mean(axis=1) / azar.std(axis=1) * np.sqrt(52)
print(tabla_bt.round(2).to_string())
print(
    "Sharpe de 1 000 carteras al azar: media",
    round(sh_azar.mean(), 2),
    "| percentil 5 y 95:",
    np.round(np.percentile(sh_azar, [5, 95]), 2),
    "| percentil de la estrategia «Conjunto»:",
    round((sh_azar < tabla_bt.loc["Conjunto", "Sharpe"]).mean() * 100, 1),
)

# Curva de capital (base 100) de la estrategia y de las referencias, y distribución del Sharpe del azar
fig, ax = plt.subplots(1, 2, figsize=(12, 3.9))
for n, c, ls in [
    ("Conjunto", "black", "-"),
    ("Logística", "black", ":"),
    ("1/N (10 activos)", "gray", "-"),
    ("SPY", "gray", "--"),
]:
    ax[0].plot((1 + semanal[n].fillna(0)).cumprod() * 100, color=c, ls=ls, label=n)
ax[0].set_title("Capital base 100 (con costos de 10 pb)")
ax[0].legend(fontsize=8)
ax[1].hist(sh_azar, bins=30, color="lightgray", edgecolor="gray")
ax[1].axvline(
    tabla_bt.loc["Conjunto", "Sharpe"],
    color="black",
    lw=2,
    label="estrategia «Conjunto»",
)
ax[1].axvline(
    tabla_bt.loc["1/N (10 activos)", "Sharpe"], color="gray", lw=2, ls="--", label="1/N"
)
ax[1].set_title("Sharpe de 1 000 carteras al azar")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig("m4_figura_1.png", dpi=130); plt.close()

# Recomendación a la última fecha con variables
ultima = panel_x.index.get_level_values("fecha").max()
hoy = panel_x.loc[ultima]
entr = panel
ajustados = {n: f().fit(entr[VARS], entr["y"]) for n, f in modelos.items()}
rec = pd.DataFrame(
    {n: m.predict_proba(hoy[VARS])[:, 1] for n, m in ajustados.items()}, index=hoy.index
)
rec["Conjunto"] = rec.mean(axis=1)
rec = rec.sort_values("Conjunto", ascending=False)
rec["señal"] = "MANTENER"
rec.loc[rec.index[:3], "señal"] = "COMPRAR"
rec.loc[rec.index[-3:], "señal"] = "EVITAR"
rec.loc[(rec["señal"] == "COMPRAR") & (rec["Conjunto"] < 0.5), "señal"] = "MANTENER"
rec["nombre"] = [NOMBRES.get(t, t) for t in rec.index]
print("Fecha de corte:", ultima.date())
print(rec[["nombre", "Logística", "XGBoost", "Conjunto", "señal"]].round(3).to_string())

# Semáforo del mercado con SPY: probabilidad de alta volatilidad (22 sesiones) y de caída extrema (10 sesiones) con el modelo de las Guías 1 y 4
c = cierres["SPY"].dropna()
a, b, v = (
    altos["SPY"].reindex(c.index),
    bajos["SPY"].reindex(c.index),
    volumenes["SPY"].reindex(c.index),
)
vx = cierres["^VIX"].reindex(c.index).ffill()
rr = np.log(c).diff() * 100
Xm = pd.DataFrame(
    {
        "ret_1": rr,
        "ret_5": rr.rolling(5).sum(),
        "ret_22": rr.rolling(22).sum(),
        "vol_5": rr.rolling(5).std(),
        "vol_22": rr.rolling(22).std(),
        "vol_66": rr.rolling(66).std(),
        "vix": vx,
        "d_vix": vx.diff(5),
        "dist_sma50": (c / c.rolling(50).mean() - 1) * 100,
        "rango_22": (c - b.rolling(22).min())
        / (a.rolling(22).max() - b.rolling(22).min()),
    }
).loc["2012-01-01":]
vol_f = rr.shift(-22).rolling(22).std()
caida = pd.concat([c.shift(-k) / c - 1 for k in range(1, 11)], axis=1).min(axis=1)
dm = Xm.join(vol_f.rename("vol_fut")).join(caida.rename("caida"))
ent_m = dm.dropna()
umbral_v = ent_m["vol_fut"].quantile(0.70)
m_v = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(
    ent_m[Xm.columns], (ent_m["vol_fut"] > umbral_v).astype(int)
)
m_c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(
    ent_m[Xm.columns], (ent_m["caida"] <= -0.04).astype(int)
)
x_hoy = Xm.iloc[[-1]]
p_vol, p_crash = float(m_v.predict_proba(x_hoy)[0, 1]), float(
    m_c.predict_proba(x_hoy)[0, 1]
)
print(
    "SPY al",
    Xm.index[-1].date(),
    "| VIX:",
    round(float(x_hoy["vix"].iloc[0]), 2),
    "| P(alta volatilidad próximas 22 sesiones):",
    round(p_vol, 3),
    "| P(caída ≥ 4 % en 10 sesiones):",
    round(p_crash, 3),
    "| tasas base históricas:",
    round(float((ent_m["vol_fut"] > umbral_v).mean()), 3),
    "y",
    round(float((ent_m["caida"] <= -0.04).mean()), 3),
)

# Exportación al tablero: JSON (con todo lo que muestra la página) y CSV de recomendaciones
equidad = {
    n: [round(float(x), 3) for x in (1 + semanal[n].fillna(0)).cumprod() * 100]
    for n in ["Conjunto", "Logística", "1/N (10 activos)", "SPY"]
}
salida = {
    "fecha_corte": str(ultima.date()),
    "portafolio": PORTAFOLIO,
    "aviso": "Material educativo; no es asesoría de inversión.",
    "semaforo": {
        "vix": round(float(x_hoy["vix"].iloc[0]), 2),
        "p_alta_volatilidad_22": round(p_vol, 3),
        "p_caida_extrema_10": round(p_crash, 3),
    },
    "recomendaciones": [
        {
            "activo": t,
            "nombre": r["nombre"],
            "p_logistica": round(float(r["Logística"]), 3),
            "p_xgboost": round(float(r["XGBoost"]), 3),
            "p_conjunto": round(float(r["Conjunto"]), 3),
            "senal": r["señal"],
        }
        for t, r in rec.iterrows()
    ],
    "backtest": {
        "fechas": [str(f.date()) for f in semanal.index],
        "equidad": equidad,
        "metricas": {
            n: {k: round(float(v), 2) for k, v in fila.items()}
            for n, fila in tabla_bt.iterrows()
        },
        "sharpe_azar": {
            "media": round(float(sh_azar.mean()), 2),
            "p5": round(float(np.percentile(sh_azar, 5)), 2),
            "p95": round(float(np.percentile(sh_azar, 95)), 2),
        },
    },
}
with open("dashboard_clasificacion.json", "w", encoding="utf-8") as f:
    json.dump(salida, f, ensure_ascii=False)
rec.round(4).to_csv("recomendaciones_5_sesiones.csv", encoding="utf-8")
print(
    "Escritos: dashboard_clasificacion.json y recomendaciones_5_sesiones.csv |",
    len(rec),
    "recomendaciones",
)