# Loan Approval Prediction Using Logistic Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
df = pd.read_csv("loan_data.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Display dataset information
print("\nDataset Information:")
print(df.info())

# Display missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove Loan_ID column (if available)
if "Loan_ID" in df.columns:
    df.drop("Loan_ID", axis=1, inplace=True)

# Fill missing values
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# Convert categorical columns into numerical values
encoder = LabelEncoder()

for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = encoder.fit_transform(df[col])

# Separate Features (X) and Target (y)
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Split the dataset into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Create and train the Logistic Regression model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Model Evaluation
print("\n========== Model Evaluation ==========\n")

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy : {accuracy*100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, labels=[0, 1], zero_division=0))

# Predict Loan Approval

sample = X.iloc[[0]]

prediction = model.predict(sample)

print("\n========== Prediction ==========\n")

if prediction[0] == 1:
    print("Loan Status : Approved")
else:
    print("Loan Status : Rejected")

# Show Prediction Probability
probability = model.predict_proba(sample)

print("\nPrediction Probability")
print(f"Rejected : {probability[0][0]*100:.2f}%")
print(f"Approved : {probability[0][1]*100:.2f}%")