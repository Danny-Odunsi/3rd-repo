# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 18:00:25 2025

@author: danny
"""

# Import libraries
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


#DATA CLEANING

# Adding the dataset
imp = pd.read_csv("Wine_data.csv", encoding="latin1") 
#print(imp)

# Removing my column notes
cleaned = imp.drop(columns=[
    "Unnamed: 12", "Unnamed: 13", "Unnamed: 14",
    "Unnamed: 15", "Unnamed: 16", "Unnamed: 17"])


#Any null values?
print(cleaned.isnull().sum())
#Fully clean


#Randomising the data for effective model learning
cleaned = shuffle(cleaned, random_state=42)  # Set a random_state for reproducibility


# FEATURE SELECTION !!!!!

# Calculate the correlation matrix for all numerical features
correlation_matrix = cleaned.corr()
print(correlation_matrix)


#GRAPH FOR CORRELATION VALUES

import matplotlib.pyplot as plt
import seaborn as sns

# Define the features and their correlation values
features = [
    'Fixed acidity', 'Volatile acidity', 'Citric acid', 'Residual sugar', 'Chlorides', 
    'Free sulfur dioxide', 'Total sulfur dioxide', 'Density', 'pH', 'Sulphates', 'Alcohol'
]
correlations = [
    0.037545, 0.020286, 0.052341, 0.049734, 0.052905, 
    0.053021, 0.035975, 0.038473, -0.002383, 0.047728, 0.081891
]

# Create a DataFrame
correlation_df = pd.DataFrame({
    'Feature': features,
    'Correlation': correlations
})

# Plot the correlation values
plt.figure(figsize=(10, 6))
sns.barplot(x='Correlation', y='Feature', data=correlation_df, palette='coolwarm')
plt.title('Correlation of Features with Wine Quality')
plt.xlabel('Correlation Value')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


# Split into features (x) and target (y)
X = cleaned.drop('quality', axis=1)
y = cleaned['quality']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# MODEL LEARNING

# Random forest classifier model 

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# Create and train the model
model = RandomForestClassifier(n_estimators=500, max_depth=30) #max_depth=10
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

"""
# Logistic Regression model 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=200000)

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
"""

"""
#Gradient boosting classifier

from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Scale the features
#scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)

# Initialize the Gradient Boosting model
model = GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_depth=10)

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
"""