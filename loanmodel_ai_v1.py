import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree

# 1. Skapa dataset
np.random.seed(42) # fast värde för slumpgeneratorn seed() som finns i numpy, ta bort denna för att få lite olika dataset, accuracy, etc varje gång

n = 500 # storlek på dataset

# fyra arrayer skapas med 500 testdata i sig, lägg märke till noise = att avslag och beviljande kan ske felaktigt
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

# train_test_split = funktion i sklearn. test_size = 0.2, vilket betyder 100 data sorteras ut innan text 
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, random_state = 42
)

# 3. Träna beslutsträd
model = DecisionTreeClassifier(max_depth = 3) # hur djupt modellen får gå 
# här används 400 slumpdata för att träna modellen rekursivt 
model.fit(x_train, y_train)

# 4. Testa modellen på osedd data (de 100 som är kvar)
predictions = model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("Confusion matrix:")
print("[kolumn 1 = 0, kolumn 2 = 1]:")
print("[post 1 = får avslag, post 2 = beviljas lån]:")
print(confusion_matrix(y_test, predictions))

#5. Visualisera trädet
plt.figure(figsize = (10, 6))
tree.plot_tree(model,
           feature_names=x.columns,
           class_names=["Denied", "Approved"],
           filled=True)
plt.show()
