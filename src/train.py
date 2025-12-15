import os
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# 1. Load Dataset (Iris)
# -------------------------------
iris = load_iris()
X = iris.data
y = iris.target

# -------------------------------
# 2. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 3. Train Model
# -------------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# 4. Evaluate Model
# -------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# -------------------------------
# 5. Save Model
# -------------------------------
os.makedirs("models", exist_ok=True)

model_path = "models/model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"Model saved at: {model_path}")
