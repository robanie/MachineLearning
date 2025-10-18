import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import joblib

# 1. load data
df = pd.read_csv("processed_kelulusan.csv")
X = df.drop("Lulus", axis=1)
y = df["Lulus"]
print("Jumlah sampel per kelas:\n", y.value_counts())

# 2. split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print("\nShapes:", X_train.shape, X_val.shape, X_test.shape)

# 3. preprocessing
num_cols = X_train.select_dtypes(include="number").columns
preprocessor = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols)
], remainder="drop")
print("\nPreprocessing ready")

# 4. logistic regression
pipe_lr = Pipeline([("pre", preprocessor), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])
pipe_lr.fit(X_train, y_train)
y_val_pred_lr = pipe_lr.predict(X_val)
print("\n=== Logistic Regression ===")
print("F1(val):", f1_score(y_val, y_val_pred_lr, average="macro"))
print(classification_report(y_val, y_val_pred_lr, digits=3))

# 5. random forest
pipe_rf = Pipeline([("pre", preprocessor), ("clf", RandomForestClassifier(n_estimators=100, max_features="sqrt", class_weight="balanced", random_state=42))])
pipe_rf.fit(X_train, y_train)
y_val_pred_rf = pipe_rf.predict(X_val)
print("\n=== Random Forest ===")
print("F1(val):", f1_score(y_val, y_val_pred_rf, average="macro"))

# 6. tuning
skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
param_grid = {"clf__max_depth": [None, 10], "clf__min_samples_split": [2, 5]}
gs = GridSearchCV(pipe_rf, param_grid=param_grid, cv=skf, scoring="f1_macro", n_jobs=-1, verbose=0)
gs.fit(X_train, y_train)
best_model = gs.best_estimator_
print("\nBest params:", gs.best_params_)
print("Best CV F1:", gs.best_score_)

# 7. val eval
y_val_best = best_model.predict(X_val)
print("\n=== Best RF (Val) ===")
print("F1(val):", f1_score(y_val, y_val_best, average="macro"))

# 8. test eval
y_test_pred = best_model.predict(X_test)
print("\n=== Test Set ===")
print("F1(test):", f1_score(y_test, y_test_pred, average="macro"))
print(classification_report(y_test, y_test_pred, digits=3))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_test_pred))

# 9. ROC curve
if hasattr(best_model, "predict_proba") and len(y_test) > 1:
    y_test_proba = best_model.predict_proba(X_test)[:,1]
    try: print("ROC-AUC(test):", roc_auc_score(y_test, y_test_proba))
    except: pass
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig("roc_test.png", dpi=120)
    plt.close()
    print("ROC saved as roc_test.png")

# 10. save model
joblib.dump(best_model, "model.pkl")
print("\nModel saved as model.pkl")