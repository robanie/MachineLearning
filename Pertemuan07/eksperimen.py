import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("processed_kelulusan.csv")
X = df.drop("Lulus", axis=1)
y = df["Lulus"]

# Normalisasi
sc = StandardScaler()
Xs = sc.fit_transform(X)

# Split pertama: train 70%, temp 30%
X_train, X_temp, y_train, y_temp = train_test_split(
    Xs,
    y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

# Split kedua: val & test 15% masing-masing, tanpa stratify
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42
)

print("Data shapes:", X_train.shape, X_val.shape, X_test.shape)

# Eksperimen
experiments = [
    {"hidden_layer_sizes": (32,16), "solver":"adam"},
    {"hidden_layer_sizes": (64,32), "solver":"adam"},
    {"hidden_layer_sizes": (128,64), "solver":"adam"},
    {"hidden_layer_sizes": (32,16), "solver":"sgd"},
    {"hidden_layer_sizes": (64,32), "solver":"sgd"},
]

for i, exp in enumerate(experiments, start=1):
    print(f"\n=== Eksperimen {i}: {exp} ===")
    
    model = MLPClassifier(
        hidden_layer_sizes=exp["hidden_layer_sizes"],
        activation='relu',
        solver=exp["solver"],
        alpha=0.001,
        batch_size=32,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=10,
        random_state=42,
        verbose=False
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    
    acc = model.score(X_test, y_test)
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {round(acc,3)} | AUC: {round(auc,3)} | F1: {round(f1,3)}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, digits=3))
    
    plt.figure()
    plt.plot