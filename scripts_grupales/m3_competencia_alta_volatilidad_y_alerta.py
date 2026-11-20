# =========================================================
# Misión 3 · Mundo 3: régimen de alta volatilidad (22 sesiones) y alerta de caída extrema de un activo del grupo
# Fase CRISP-DM: Modelado · Evaluación | Datos: REALES (Yahoo Finance) | Semilla global: 2026
# Uso: python m3_competencia_alta_volatilidad_y_alerta.py TICKER   (por defecto QQQ)
# =========================================================
import sys, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss, recall_score, precision_score
from xgboost import XGBClassifier
from comun_m13 import cargar, variables

warnings.filterwarnings("ignore")
SEMILLA, H = 2026, 22
TICKER = sys.argv[1] if len(sys.argv) > 1 else "QQQ"
c, a, b, v = cargar(extra=[TICKER])
cc, r, X = variables(c, a, b, v, TICKER, c["^VIX"])
VARS = list(X.columns)
vol_f = r.shift(-H).rolling(H).std()
caida = pd.concat([cc.shift(-k) / cc - 1 for k in range(1, 11)], axis=1).min(axis=1)
d = X.join(vol_f.rename("vol_fut")).join(caida.rename("caida")).dropna()
k = int(len(d) * 0.7)
ent, pru = d.iloc[: k - H].copy(), d.iloc[k:].copy()
umbral = ent["vol_fut"].quantile(0.70)
for t in (ent, pru):
    t["alta_vol"] = (t["vol_fut"] > umbral).astype(int)
    t["crash"] = (t["caida"] <= -0.04).astype(int)
print(TICKER, "| entrenamiento:", len(ent), "hasta", ent.index.max().date(), "| prueba:", len(pru), "desde", pru.index.min().date())
print("Tasa base de alta volatilidad (prueba):", round(pru["alta_vol"].mean(), 3), "| de caída extrema (prueba):", round(pru["crash"].mean(), 3))
tscv = TimeSeriesSplit(n_splits=4, gap=H)
cand = {"Logística": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), {"logisticregression__C": np.logspace(-3, 1, 20)}),
        "Naive Bayes": (GaussianNB(), {"var_smoothing": np.logspace(-10, -6, 5)}),
        "SVM RBF": (make_pipeline(StandardScaler(), SVC(probability=True, random_state=SEMILLA)), {"svc__C": [0.1, 1, 10], "svc__gamma": [0.01, 0.03, 0.1]}),
        "Random Forest": (RandomForestClassifier(random_state=SEMILLA, n_jobs=-1), {"n_estimators": [200, 400], "max_depth": [3, 5, 8], "min_samples_leaf": [10, 30, 60]}),
        "XGBoost": (XGBClassifier(random_state=SEMILLA, n_jobs=1), {"n_estimators": [150, 300], "learning_rate": [0.03, 0.1], "max_depth": [2, 3, 4], "min_child_weight": [5, 20]})}
for objetivo in ["alta_vol", "crash"]:
    filas, mejores = [], {}
    for nombre, (m, rej) in cand.items():
        n_iter = min(8, int(np.prod([len(x) for x in rej.values()])))
        bus = RandomizedSearchCV(m, rej, n_iter=n_iter, cv=tscv, scoring="roc_auc", random_state=SEMILLA, n_jobs=1).fit(ent[VARS], ent[objetivo])
        mejores[nombre] = bus.best_estimator_
        p = bus.predict_proba(pru[VARS])[:, 1]
        filas.append([nombre, bus.best_score_, roc_auc_score(pru[objetivo], p), average_precision_score(pru[objetivo], p),
                      f1_score(pru[objetivo], (p >= 0.5).astype(int)), brier_score_loss(pru[objetivo], p)])
    tabla = pd.DataFrame(filas, columns=["modelo", "AUC_validacion", "AUC_prueba", "AUC_PR_prueba", "F1_prueba", "Brier_prueba"]).sort_values("AUC_prueba", ascending=False)
    print(f"\nObjetivo: {objetivo} | tasa base en la prueba {pru[objetivo].mean():.3f}")
    print(tabla.round(3).to_string(index=False))
    mejor = tabla.iloc[0]["modelo"]
    perm = permutation_importance(mejores[mejor], pru[VARS], pru[objetivo], n_repeats=20, random_state=SEMILLA, scoring="roc_auc")
    print(f"Mejor modelo: {mejor} | importancia por permutación (caída del AUC):")
    print(pd.Series(perm.importances_mean, index=VARS).sort_values(ascending=False).round(4).head(5).to_string())
    if objetivo == "crash":                                  # umbral por costos elegido en validación (no en la prueba)
        k2 = int(len(ent) * 0.8)
        e2, va = ent.iloc[: k2 - H], ent.iloc[k2:]
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(e2[VARS], e2["crash"])
        pv = m.predict_proba(va[VARS])[:, 1]
        us, c_fp, c_fn = np.linspace(0.03, 0.9, 88), 1.0, 10.0
        costo = [c_fp * ((pv >= u) & (va["crash"] == 0)).sum() + c_fn * ((pv < u) & (va["crash"] == 1)).sum() for u in us]
        u_opt = us[int(np.argmin(costo))]
        pp = m.fit(ent[VARS], ent["crash"]).predict_proba(pru[VARS])[:, 1]
        yh = (pp >= u_opt).astype(int)
        cp = c_fp * ((yh == 1) & (pru["crash"] == 0)).sum() + c_fn * ((yh == 0) & (pru["crash"] == 1)).sum()
        print(f"Umbral por costos (FN = {c_fn:.0f}, FP = {c_fp:.0f}) elegido en validación: {u_opt:.2f} | exhaustividad {recall_score(pru['crash'], yh):.3f}",
              f"| precisión {precision_score(pru['crash'], yh, zero_division=0):.3f} | costo en la prueba {cp:.0f} (nunca avisar: {c_fn * pru['crash'].sum():.0f})")
