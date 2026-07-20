# Student Pass/Fail Prediction Using Logistic Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset
df = pd.read_csv("student_data.csv")

# Features
X = df[["StudyHours", "Attendance"]]

# Target
y = df["Pass"]

# Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model

model = LogisticRegression()

model.fit(X_train, y_train)

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("Accuracy:", accuracy)

print(classification_report(y_test, prediction))

cm = confusion_matrix(y_test, prediction)

sns.heatmap(cm, annot=True, cmap="Greens", fmt="d")

plt.title("Student Pass Prediction")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

sample = [[8,92]]

result = model.predict(sample)

if result[0] == 1:
    print("Student Will Pass")
else:
    print("Student Will Fail")