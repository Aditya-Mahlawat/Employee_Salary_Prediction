import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
print("Loading dataset...")
data = pd.read_csv("adult 3.csv")

# Data cleaning (similar to the notebook)
print("Cleaning data...")
# Handle missing values
data.workclass.replace({'?':'Others'}, inplace=True)
data.occupation.replace({'?':'Others'}, inplace=True)

# Remove outliers and unnecessary categories
data = data[(data['age'] <= 75) & (data['age'] >= 17)]
data = data[data['workclass'] != 'Without-pay']
data = data[data['workclass'] != 'Never-worked']
data = data[data['education'] != '1st-4th']
data = data[data['education'] != '5th-6th']
data = data[data['education'] != 'Preschool']

# Drop education column as educational-num provides similar information
data.drop(columns=['education'], inplace=True)

# Encode categorical features
print("Encoding categorical features...")
encoder = LabelEncoder()
categorical_cols = ['workclass', 'marital-status', 'occupation', 
                   'relationship', 'race', 'gender', 'native-country']

for col in categorical_cols:
    data[col] = encoder.fit_transform(data[col])

# Split into features and target
X = data.drop(columns=['income'])
Y = data['income']

# Scale features
print("Scaling features...")
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split into training and testing sets
print("Splitting data...")
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=23, stratify=Y)

# Train the model
print("Training model...")
knn = KNeighborsClassifier()
knn.fit(xtrain, ytrain)

# Evaluate the model
predictions = knn.predict(xtest)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(ytest, predictions)
print(f"Model accuracy: {accuracy:.4f}")

# Save the model
print("Saving model...")
joblib.dump(knn, "best_model.pkl")
print("Model saved successfully!")