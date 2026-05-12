import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import joblib

np.random.seed(42)

n = 500

income = np.random.normal(30000, 8000, n)
age = np.random.normal(40, 10, n)
credit_score = np.random.normal(600, 100, n)
noise = np.random.normal(0, 2000, n)

loan_approved = (
    (income + noise > 28000) &
    (credit_score > 550) &
    (age > 25)
).astype(int)

data = pd.DataFrame({
    "income": income,
    "age": age,
    "credit_score": credit_score,
    "loan_approved": loan_approved
})

x = data[["income", "age", "credit_score"]]
y = data["loan_approved"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

joblib.dump(model, "compare.pkl")