import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

# --- STEP 2: LOADING AND EXPLORING ---

# 1. Load the file (Ensure creditcard.csv is in your project folder!)
data = pd.read_csv("creditcard.csv")

# 2. Preview the data (The first 5 rows)
print("--- DATA PREVIEW ---")
print(data.head())

# 3. Statistical Summary
# This gives you the mean, min, max, and standard deviation for every column
print("\n--- STATISTICAL SUMMARY ---")
print(data.describe())

data = pd.read_csv("creditcard.csv")
print(data.head())

data = pd.read_csv("creditcard.csv")
print(data.head())

fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
outlierFraction = len(fraud)/float(len(valid))
print(outlierFraction)
print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))
print('Valid Transactions: {}'.format(len(data[data['Class'] == 0])))

# --- STEP 4: EXPLORING TRANSACTION AMOUNTS ---

print("\n--- AMOUNT DETAILS: FRAUDULENT TRANSACTIONS ---")
print(fraud.Amount.describe())

print("\n--- AMOUNT DETAILS: VALID TRANSACTIONS ---")
print(valid.Amount.describe())

# --- STEP 5: PLOTTING CORRELATION MATRIX ---

# 1. Calculate the correlations
corrmat = data.corr()

# 2. Set the size of the plot
fig = plt.figure(figsize = (12, 9))

# 3. Create the heatmap
# 'vmax' controls the color intensity
sns.heatmap(corrmat, vmax = .8, square = True)

# 4. Show the plot
plt.show()

# --- STEP 6: PREPARING DATA ---

X = data.drop(['Class'], axis = 1)
Y = data["Class"]
print(X.shape)
print(Y.shape)

xData = X.values
yData = Y.values

from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(
        xData, yData, test_size = 0.2, random_state = 42)

# --- STEP 7: BUILDING AND TRAINING THE MODEL ---
from sklearn.ensemble import RandomForestClassifier

print("\n--- STARTING MODEL TRAINING ---")
print("This may take 1-2 minutes. Please wait...")

# 1. Initialize and Train
rfc = RandomForestClassifier()
rfc.fit(xTrain, yTrain)

# 2. Make Predictions
yPred = rfc.predict(xTest)
print("Training and Predictions complete!")


# --- STEP 8: EVALUATING THE MODEL ---
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, matthews_corrcoef, confusion_matrix)

# Calculate Metrics
acc = accuracy_score(yTest, yPred)
prec = precision_score(yTest, yPred)
rec = recall_score(yTest, yPred)
f1 = f1_score(yTest, yPred)
mcc = matthews_corrcoef(yTest, yPred)

print("\n--- MODEL EVALUATION METRICS ---")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Matthews Correlation Coefficient: {mcc:.4f}")

# Plotting the Confusion Matrix
conf_matrix = confusion_matrix(yTest, yPred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()

