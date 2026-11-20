# =========================================================
# Misión 1 · ¿Se puede predecir la dirección a 5 sesiones de los activos del grupo? (Mundo 1)
# Fase CRISP-DM: Modelado · Evaluación | Semilla global: 2026 | Datos: REALES (Yahoo Finance)
# Uso: python m1_direccion_portafolio_prueba_de_azar.py BVN,SCCO,AAPL,JPM,KO [horizonte]   (por defecto, el portafolio de referencia y 5)
# =========================================================
import sys, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score
from xgboost import XGBClassifier
from comun_m13 import cargar, variables

warnings.filterwarnings("ignore")
SEMILLA = 2026
rng = np.random.default_rng(SEMILLA)
H = int(sys.argv[2]) if len(sys.argv) > 2 else 5                                     # horizonte en sesiones
ACTIVOS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["BVN", "FSM", "SCCO", "BHP", "B", "AAPL", "JPM", "XOM", "KO", "PFE"]
c, a, b, v = cargar(extra=ACTIVOS)
filas = []
for t in ACTIVOS:
    cc, r, X = variables(c, a, b, v, t, c["^VIX"])
    y = (cc.shift(-H) / cc > 1).astype(float)
    d = X.join(y.rename("y")).dropna()
    VARS = list(X.columns)
    k = int(len(d) * 0.7)
    ent, pru = d.iloc[: k - H], d.iloc[k:]
    modelos = {"Logística": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), "Naive Bayes": GaussianNB(),
               "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=30, random_state=SEMILLA, n_jobs=-1),
               "XGBoost": XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=2, subsample=0.7, min_child_weight=30, random_state=SEMILLA, n_jobs=1)}
    for nombre, m in modelos.items():
        p = m.fit(ent[VARS], ent["y"]).predict_proba(pru[VARS])[:, 1]
        auc = roc_auc_score(pru["y"], p)
        azar = np.array([roc_auc_score(rng.permutation(pru["y"].values), p) for _ in range(500)])
        filas.append([t, nombre, pru["y"].mean(), accuracy_score(pru["y"], (p >= 0.5).astype(int)),
                      balanced_accuracy_score(pru["y"], (p >= 0.5).astype(int)), auc, (azar >= auc).mean()])
res = pd.DataFrame(filas, columns=["activo", "modelo", "tasa_base", "exactitud", "exactitud_balanceada", "AUC", "valor_p"])
print(f"Horizonte: {H} sesiones | prueba = 30 % final de cada serie (embargo de {H})")
print(res.round(3).to_string(index=False))
print("\nAUC medio por modelo:")
print(res.groupby("modelo")["AUC"].mean().round(3).to_string())
n_sig = int((res["valor_p"] < 0.05).sum())
print(f"\nPruebas con valor p < 0,05: {n_sig} de {len(res)} (por azar se esperan {0.05 * len(res):.1f})")
print("Exactitud media de los modelos:", round(res["exactitud"].mean(), 3), "| tasa base media (exactitud de «siempre sube»):", round(res["tasa_base"].mean(), 3))
res.to_csv(f"m1_resultados_h{H}.csv", index=False)
