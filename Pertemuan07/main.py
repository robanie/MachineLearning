# 1. Import library
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 2. Load data (contoh pakai dataset iris)
from sklearn.datasets import load_iris
data = load_iris()

# Pisahkan fitur dan target
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# 3. Training dan testing split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 4. Training model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 5. Evaluasi model
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Akurasi Model:", acc)