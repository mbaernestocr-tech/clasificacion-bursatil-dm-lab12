# =========================================================
# Misión 2 · Mundo 2: ¿qué métrica engaña cuando el evento es raro? (sintético, semilla fija)
# Fase CRISP-DM: Modelado · Evaluación | Datos: SINTÉTICOS (make_classification)
# =========================================================
import warnings
import numpy as np, pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, recall_score, average_precision_score, roc_auc_score, log_loss
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as PipelineIB

warnings.filterwarnings("ignore")
SEMILLAS = [2026, 2027, 2028, 2029, 2030]
# Barrido 1: la tasa base del evento frente a las métricas
filas = []
for tb in [0.5, 0.2, 0.05, 0.01, 0.002]:
    for s in SEMILLAS:
        X, y = make_classification(n_samples=40000, n_features=10, n_informative=4, n_redundant=2, weights=[1 - tb, tb], class_sep=1.0, flip_y=0, random_state=s)
        Xe, Xp, ye, yp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=s)
        p = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xe, ye).predict_proba(Xp)[:, 1]
        yh = (p >= 0.5).astype(int)
        filas.append([tb, s, 1 - yp.mean(), accuracy_score(yp, yh), recall_score(yp, yh), f1_score(yp, yh, zero_division=0), average_precision_score(yp, p), roc_auc_score(yp, p)])
b1 = pd.DataFrame(filas, columns=["tasa_base", "semilla", "exactitud_siempre_no", "exactitud", "exhaustividad", "F1", "AUC_PR", "AUC_ROC"])
print("Barrido 1 (media de 5 semillas): la tasa base del evento frente a las métricas")
print(b1.groupby("tasa_base").mean().drop(columns="semilla").round(3).to_string())
print("Desviación estándar del AUC_PR:", b1.groupby("tasa_base")["AUC_PR"].std().round(3).to_dict())
# Barrido 2: SMOTE antes de dividir (fuga) frente a SMOTE dentro del pipeline
filas = []
for s in SEMILLAS:
    X, y = make_classification(n_samples=6000, n_features=12, n_informative=3, n_redundant=0, weights=[0.98, 0.02], class_sep=0.6, flip_y=0.02, random_state=s)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
    rf = RandomForestClassifier(n_estimators=150, min_samples_leaf=3, random_state=s, n_jobs=-1)
    Xm, ym = SMOTE(random_state=s).fit_resample(X, y)
    filas.append([s, cross_val_score(rf, Xm, ym, cv=cv, scoring="roc_auc").mean(),
                  cross_val_score(PipelineIB([("smote", SMOTE(random_state=s)), ("rf", rf)]), X, y, cv=cv, scoring="roc_auc").mean(),
                  cross_val_score(rf, X, y, cv=cv, scoring="roc_auc").mean()])
b2 = pd.DataFrame(filas, columns=["semilla", "AUC_SMOTE_antes", "AUC_SMOTE_dentro", "AUC_sin_remuestreo"])
print("\nBarrido 2: SMOTE antes de dividir frente a SMOTE dentro del pipeline (AUC en validación cruzada)")
print(b2.round(3).to_string(index=False))
print("Media:", b2.drop(columns="semilla").mean().round(3).to_dict())
# Barrido 3: Naive Bayes con copias de una variable (sobreconfianza)
filas = []
for s in SEMILLAS:
    rg = np.random.default_rng(s)
    n = 6000
    y = rg.binomial(1, 0.3, n)
    x1, x2 = rg.normal(0.8 * y, 1), rg.normal(0.8 * y, 1)
    for copias in [0, 10]:
        extra = np.column_stack([x1 + rg.normal(0, 0.3, n) for _ in range(copias)]) if copias else np.empty((n, 0))
        Xd = np.column_stack([x1, x2, extra])
        Xe, Xp, ye, yp = train_test_split(Xd, y, test_size=0.4, random_state=s, stratify=y)
        for nombre, m in [("Naive Bayes", GaussianNB()), ("Logística", LogisticRegression(max_iter=2000))]:
            p = np.clip(m.fit(Xe, ye).predict_proba(Xp)[:, 1], 1e-6, 1 - 1e-6)
            filas.append([s, copias, nombre, roc_auc_score(yp, p), log_loss(yp, p)])
b3 = pd.DataFrame(filas, columns=["semilla", "copias", "modelo", "AUC", "log_loss"])
print("\nBarrido 3: copias de una variable (media de 5 semillas)")
print(b3.groupby(["copias", "modelo"])[["AUC", "log_loss"]].mean().round(3).to_string())
b1.to_csv("m2_barrido1_tasa_base.csv", index=False)
b2.to_csv("m2_barrido2_smote.csv", index=False)
b3.to_csv("m2_barrido3_naive_bayes.csv", index=False)
