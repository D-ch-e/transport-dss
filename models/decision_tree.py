# Decision_Tree_System.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

path = r'/content/drive/MyDrive/ML_project/content/sample_data/synthetic_data/720_500_model_data_m1.csv'
data_tree = pd.read_csv(path)
best_data = data_tree[data_tree["label"] == "best"].copy()

def get_best_mode(row):

    if row["mode_Bus"] == 1:
        return "Bus"

    elif row["mode_Car"] == 1:
        return "Car"

    elif row["mode_HSR"] == 1:
        return "HSR"

    elif row["mode_TRA"] == 1:
        return "TRA"

best_data["best_mode"] = best_data.apply(
    get_best_mode,
    axis=1
)

# ------ model training ------
X_clean = best_data[[
    "people", "luggage", "budget", "time_pref", "cost_pref", "conv_pref","distance_km"
]] # Only keep user-related features

y_clean = best_data["best_mode"]

X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.2, random_state=42
)

clf = DecisionTreeClassifier(random_state=42) # Decision Tree
clf.fit(X_train, y_train)

# ------ model evaluation ------
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
