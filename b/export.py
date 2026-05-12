import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree
import joblib

np.random.seed(42)

n = 500

income = np.random.normal(30000, 8000, n)
age = np.random.normal(40, 10, n)
credit_score = np.random.normal(600, 100, n)
noise = np.random.normal(0, 2000, n)

loan_approved = (
    (income + noise > 28000) &
    (credit_score > 550)
).astype(int)

data = pd.DataFrame({
    "income": income,
    "age": age,
    "credit_score": credit_score,
    "loan_approved": loan_approved
})

# print(data)

# 2. Dela upp data
x = data[["income", "age", "credit_score"]]
y = data["loan_approved"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, random_state = 42
)

model = DecisionTreeClassifier(max_depth = 3)
model.fit(x_train, y_train)

predictions = model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("Confusion matrix:")
print("[kolumn 1 = 0, kolumn 2 = 1]:")
print("[post 1 = får avslag, post 2 = beviljas lån]:")
print(confusion_matrix(y_test, predictions))

# plt.figure(figsize = (10, 6))
# tree.plot_tree(model,
#            feature_names=x.columns,
#            class_names=["Denied", "Approved"],
#            filled=True)
# plt.show()