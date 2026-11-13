from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parent)

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

import warnings
warnings.filterwarnings('ignore')

#uploading the dataset
df = pd.read_csv('heart_2022_no_nans.csv')

print(# checking first five rows of the dataset
df.head())

print(# checking shape of the dataset
df.shape)

# checking information of teh dataset
df.info()

print(# descriptive statistics
df.describe())

print(# checking null values
df.isnull().sum())


print("\n=== Data Visualisation ===")



plt.figure(figsize=(6,5))
sns.countplot(data=df, x='HadHeartAttack', palette='Set2')

plt.title('Distribution of Heart Attack Cases')
plt.xlabel('Heart Attack')
plt.ylabel('Number of Patients')

plt.show()

plt.figure(figsize=(12,6))

sns.countplot(
    data=df,
    x='AgeCategory',
    hue='HadHeartAttack',
    palette='Set1'
)

plt.title('Heart Attack Distribution across Age Categories')
plt.xlabel('Age Category')
plt.ylabel('Count')
plt.xticks(rotation=45)

plt.show()

plt.figure(figsize=(6,5))

sns.countplot(
    data=df,
    x='Sex',
    hue='HadHeartAttack',
    palette='husl'
)

plt.title('Heart Attack Cases by Sex')
plt.xlabel('Sex')
plt.ylabel('Count')

plt.show()

plt.figure(figsize=(8,5))

sns.histplot(
    data=df,
    x='BMI',
    bins=30,
    kde=True,
    color='steelblue'
)

plt.title('Distribution of Body Mass Index (BMI)')
plt.xlabel('BMI')
plt.ylabel('Frequency')

plt.show()

plt.figure(figsize=(12,6))

sns.countplot(
    data=df,
    x='SmokerStatus',
    hue='HadHeartAttack',
    palette='Dark2'
)

plt.title('Smoking Status and Heart Attack')
plt.xlabel('Smoking Status')
plt.ylabel('Count')

plt.xticks(rotation=30)

plt.show()

plt.figure(figsize=(10,6))

sns.countplot(
    data=df,
    x='GeneralHealth',
    hue='HadHeartAttack',
    palette='viridis'
)

plt.title('General Health and Heart Attack')
plt.xlabel('General Health')
plt.ylabel('Count')

plt.xticks(rotation=20)

plt.show()

plt.figure(figsize=(7,5))

sns.countplot(
    data=df,
    x='PhysicalActivities',
    hue='HadHeartAttack',
    palette='Set3'
)

plt.title('Physical Activity and Heart Attack')
plt.xlabel('Physical Activities')
plt.ylabel('Count')

plt.show()

from sklearn.preprocessing import LabelEncoder

# Encode all categorical columns
label_encoder = LabelEncoder()

for column in df.select_dtypes(include='object').columns:
    df[column] = label_encoder.fit_transform(df[column])

# Verify data types
df.info()

# ---------------------------------------------------------
# Correlation Heatmap
# ---------------------------------------------------------

plt.figure(figsize=(25,20))

sns.heatmap(
    df.corr(),
    annot=True,
    fmt='.2f',
    cmap='RdYlBu',
    linewidths=0.5
)

plt.title('Correlation Matrix')
plt.show()


print("\n=== Model Implementation and Evaluation ===")

X = df.drop('HadHeartAttack', axis=1)
y = df['HadHeartAttack']

from sklearn.model_selection import train_test_split

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.125,
    random_state=42,
    stratify=y_train_val
)

print("Training set:", X_train.shape, y_train.shape)
print("Validation set:", X_val.shape, y_val.shape)
print("Testing set:", X_test.shape, y_test.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


print("\n=== Model Training ===")

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train_scaled, y_train)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)


print("\n=== Validation Set Evaluation ===")

from sklearn.metrics import roc_auc_score

validation_models = {
    'Logistic Regression': (lr, X_val_scaled),
    'Decision Tree': (dt, X_val),
    'Random Forest': (rf, X_val)
}

validation_results = []

for model_name, (model, X_validation) in validation_models.items():
    y_val_pred = model.predict(X_validation)
    y_val_prob = model.predict_proba(X_validation)[:, 1]
    validation_results.append({
        'Model': model_name,
        'Accuracy': accuracy_score(y_val, y_val_pred),
        'Precision': precision_score(y_val, y_val_pred, zero_division=0),
        'Recall': recall_score(y_val, y_val_pred, zero_division=0),
        'F1-Score': f1_score(y_val, y_val_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_val, y_val_prob)
    })

validation_results = pd.DataFrame(validation_results)
validation_results.round(4)


print("\n=== Testing Set Evaluation ===")

y_pred_lr = lr.predict(X_test_scaled)

print("========== Logistic Regression ==========\n")

print("Accuracy :", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr))
print("Recall   :", recall_score(y_test, y_pred_lr))
print("F1-Score :", f1_score(y_test, y_pred_lr))

print("\nClassification Report\n")
print(classification_report(y_test, y_pred_lr))

cm = confusion_matrix(y_test, y_pred_lr)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['No Heart Attack','Heart Attack'],
    yticklabels=['No Heart Attack','Heart Attack']
)

plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

y_pred_dt = dt.predict(X_test)

print("========== Decision Tree ==========\n")

print("Accuracy :", accuracy_score(y_test, y_pred_dt))
print("Precision:", precision_score(y_test, y_pred_dt))
print("Recall   :", recall_score(y_test, y_pred_dt))
print("F1-Score :", f1_score(y_test, y_pred_dt))

print("\nClassification Report\n")
print(classification_report(y_test, y_pred_dt))

cm = confusion_matrix(y_test, y_pred_dt)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=['No Heart Attack','Heart Attack'],
    yticklabels=['No Heart Attack','Heart Attack']
)

plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

y_pred_rf = rf.predict(X_test)

print("========== Random Forest ==========\n")

print("Accuracy :", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall   :", recall_score(y_test, y_pred_rf))
print("F1-Score :", f1_score(y_test, y_pred_rf))

print("\nClassification Report\n")
print(classification_report(y_test, y_pred_rf))

cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Purples',
    xticklabels=['No Heart Attack','Heart Attack'],
    yticklabels=['No Heart Attack','Heart Attack']
)

plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

results = pd.DataFrame({
    'Model': [
        'Logistic Regression',
        'Decision Tree',
        'Random Forest'
    ],

    'Accuracy': [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_dt),
        accuracy_score(y_test, y_pred_rf)
    ],

    'Precision': [
        precision_score(y_test, y_pred_lr),
        precision_score(y_test, y_pred_dt),
        precision_score(y_test, y_pred_rf)
    ],

    'Recall': [
        recall_score(y_test, y_pred_lr),
        recall_score(y_test, y_pred_dt),
        recall_score(y_test, y_pred_rf)
    ],

    'F1-Score': [
        f1_score(y_test, y_pred_lr),
        f1_score(y_test, y_pred_dt),
        f1_score(y_test, y_pred_rf)
    ]
})

results



results.set_index('Model').plot(
    kind='bar',
    figsize=(12,6),
    width=0.8
)

plt.title('Performance Comparison of Machine Learning Models')
plt.xlabel('Machine Learning Models')
plt.ylabel('Score')
plt.xticks(rotation=20)
plt.ylim(0,1.05)
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Predicted probabilities
lr_prob = lr.predict_proba(X_test_scaled)[:, 1]
dt_prob = dt.predict_proba(X_test)[:, 1]
rf_prob = rf.predict_proba(X_test)[:, 1]

# ROC Curve and AUC
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob)
auc_lr = roc_auc_score(y_test, lr_prob)

fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_prob)
auc_dt = roc_auc_score(y_test, dt_prob)

fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
auc_rf = roc_auc_score(y_test, rf_prob)

# Plot ROC Curves
plt.figure(figsize=(8,6))

plt.plot(fpr_lr, tpr_lr, linewidth=2,
         label=f'Logistic Regression (AUC = {auc_lr:.3f})')

plt.plot(fpr_dt, tpr_dt, linewidth=2,
         label=f'Decision Tree (AUC = {auc_dt:.3f})')

plt.plot(fpr_rf, tpr_rf, linewidth=2,
         label=f'Random Forest (AUC = {auc_rf:.3f})')

# Random classifier reference line
plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')

plt.title('ROC Curves for Logistic Regression, Decision Tree and Random Forest')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(loc='lower right')
plt.grid(True)

plt.show()

