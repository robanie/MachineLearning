import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import joblib

# Baca dataset
df = pd.read_csv("processed_kelulusan.csv")  # sesuaikan path CSV

X = df.drop("Lulus", axis=1)
y = df["Lulus"]

# Split pertama: train 70%, temp 30% (gunakan stratify)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Split kedua: val/test (tanpa stratify karena dataset kecil)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Pipeline preprocessing
num_cols = X_train.select_dtypes(include="number").columns
pre = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ]), num_cols)
], remainder="drop")

# RandomForest baseline
rf = RandomForestClassifier(
    n_estimators=300, max_features="sqrt",
    class_weight="balanced", random_state=42
)
pipe = Pipeline([("pre", pre), ("clf", rf)])
pipe.fit(X_train, y_train)

# Evaluasi validation
y_val_pred = pipe.predict(X_val)
print("F1(val):", f1_score(y_val, y_val_pred, average="macro"))
print(classification_report(y_val, y_val_pred, digits=3))

# Cross-validation
skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="f1_macro", n_jobs=-1)
print("CV F1-macro:", scores.mean(), "±", scores.std())

# GridSearchCV tuning
param = {"clf__max_depth": [None, 12, 20, 30], "clf__min_samples_split": [2, 5, 10]}
gs = GridSearchCV(pipe, param_grid=param, cv=skf, scoring="f1_macro", n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)
best_model = gs.best_estimator_
y_val_best = best_model.predict(X_val)
print("Best F1(val):", f1_score(y_val, y_val_best, average="macro"))

# Evaluasi test
y_test_pred = best_model.predict(X_test)
print("F1(test):", f1_score(y_test, y_test_pred, average="macro"))
print(classification_report(y_test, y_test_pred, digits=3))
print("Confusion Matrix:", confusion_matrix(y_test, y_test_pred))

# ROC & Precision-Recall dengan simpan otomatis
if hasattr(best_model, "predict_proba"):
    y_proba = best_model.predict_proba(X_test)[:,1]
    try:
        print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    except:
        pass

    # ROC Curve
    fig1 = plt.figure()
    plt.plot(roc_curve(y_test, y_proba)[0], roc_curve(y_test, y_proba)[1])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    fig1.tight_layout()
    fig1.savefig("roc_test.png", dpi=120)
    plt.show()

    # Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    fig2 = plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")
    fig2.tight_layout()
    fig2.savefig("pr_test.png", dpi=120)
    plt.show()

# Feature importance
try:
    importances = best_model.named_steps["clf"].feature_importances_
    fn = best_model.named_steps["pre"].get_feature_names_out()
    top = sorted(zip(fn, importances), key=lambda x: x[1], reverse=True)
    print("Top features:")
    for name, val in top[:10]:
        print(f"{name}: {val:.4f}")
except:
    print("Feature importance tidak tersedia")

# Simpan model
joblib.dump(best_model, "rf_model.pkl")
print("Model disimpan sebagai rf_model.pkl")

# Contoh prediksi
mdl = joblib.load("rf_model.pkl")
sample = pd.DataFrame([{
    "IPK": 3.4,
    "Jumlah_Absensi": 4,
    "Waktu_Belajar_Jam": 7,
    "Rasio_Absensi": 4/14,
    "IPK_x_Study": 3.4*7
}])
print("Prediksi sample:", int(mdl.predict(sample)[0]))