# ======================================================
# CUSTOMER CHURN PREDICTION AND RETENTION STRATEGY
# ======================================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
   
    roc_auc_score,
    roc_curve
)

# ======================================================
# TASK 103 - LOAD DATASET & CHURN RATE
# ======================================================

# Load Dataset
file_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(file_path)

print("\n========== DATASET PREVIEW ==========")
print(df.head())

print("\n========== DATASET INFO ==========")
print(df.info())

# Calculate Overall Churn Rate
churn_rate = (df['Churn'].value_counts()['Yes'] / len(df)) * 100

print(f"\nOverall Churn Rate: {churn_rate:.2f}%")

# ======================================================
# TASK 104 - PREPROCESSING
# ======================================================

# Convert  to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing values
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Drop customerID column
df.drop('customerID', axis=1, inplace=True)

# Encode categorical columns
label_encoders = {}

for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column]) #text to clomn
        label_encoders[column] = le

# Separate Features and Target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Normalize numeric columns
numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']

scaler = StandardScaler()
X[numeric_columns] = scaler.fit_transform(X[numeric_columns]) ## usees z transform

print("\n========== PREPROCESSED DATA ==========")
print(X.head())

# ======================================================
# TASK 105 - TRAIN DECISION TREE CLASSIFIER
# ======================================================

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create Decision Tree Model
model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

# Train Model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  

# ======================================================
# TASK 106 - MODEL EVALUATION
# ======================================================

# Accuracy
accuracy = accuracy_score(y_test, y_pred) # ytest = actual values

print(f"\nAccuracy: {accuracy:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\n========== CONFUSION MATRIX ==========")
print(cm)

# Classification Report
print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_pred))

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_prob)

print(f"ROC-AUC Score: {roc_auc:.4f}")

# ======================================================
# CONFUSION MATRIX VISUALIZATION
# ======================================================

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['No Churn', 'Churn'],
    yticklabels=['No Churn', 'Churn']
)

plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout() ##Automatically adjusts spacing

plt.savefig('churn_confusion_matrix.png')

plt.close()

# ======================================================
# ROC CURVE
# ======================================================

fpr, tpr, thresholds =  roc_curve(y_test, y_prob)

plt.figure(figsize=(7, 6))

plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.title('ROC Curve')
plt.legend()

plt.tight_layout()

plt.savefig('churn_roc_curve.png')

plt.close()

# ======================================================
# TASK 107 - DECISION TREE VISUALIZATION
# ======================================================

plt.figure(figsize=(24, 12))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=['No Churn', 'Churn'],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title('Decision Tree Visualization')

plt.savefig('decision_tree.png')

plt.close()

# ======================================================
# FEATURE IMPORTANCE
# ======================================================

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")
print(feature_importance.head(10))

# Top 3 Features
top_features = feature_importance.head(3)

print("\nTop 3 Churn Driving Features:")
print(top_features)


# ======================================================
# TASK 108 - RETENTION STRATEGIES
# ======================================================

# ======================================================
# DYNAMIC RETENTION STRATEGIES
# ======================================================

strategy_map = {
    'tenure': 'Reward loyal customers with exclusive benefits.',
    
    'MonthlyCharges': 'Provide discounted pricing and flexible plans.',
    
    'Contract': 'Encourage customers to switch to long-term contracts.',
    
    'PaymentMethod': 'Offer cashback and easier payment options.',
    
    'TechSupport': 'Improve technical support and customer service.',
    
    'InternetService': 'Provide better internet packages and speed upgrades.',
    
    'OnlineSecurity': 'Offer free security add-ons for customers.',
    
    'StreamingTV': 'Bundle entertainment subscription offers.',
    
    'TotalCharges': 'Provide personalized billing discounts.'
}

top_features = feature_importance.head(3)

strategies_list = []

for feature in top_features['Feature']:
    
    strategy = strategy_map.get(
        feature,
        'Provide personalized retention offers.'
    )
    
    strategies_list.append(strategy)

strategies = pd.DataFrame({
    'Churn Driver': top_features['Feature'].values,
    'Retention Strategy': strategies_list
})

print("\n========== RETENTION STRATEGIES ==========")
print(strategies)

strategies.to_csv('retention_strategies.csv', index=False)

# ======================================================
# END OF PROJECT
# ======================================================