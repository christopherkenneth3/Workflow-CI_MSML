import os
import pandas as pd
import kagglehub
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ======================
# MLflow CONFIG
# ======================


# AKTIFKAN AUTOLOG
mlflow.sklearn.autolog()

# ======================
# LOAD DATASET
# ======================
path = kagglehub.dataset_download("uciml/glass")
df = pd.read_csv(os.path.join(path, "glass.csv"))


# ======================
# SPLIT DATA
# ======================
X = df.drop('Type', axis=1)
y = df['Type']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ======================
# SCALING
# ======================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ======================
# TRAIN MODEL
# ======================
with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))