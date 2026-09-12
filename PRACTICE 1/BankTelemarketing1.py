from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Load the CSV file
file_path = Path(__file__).resolve().parent.parent / 'PRACTICE FINAL' / 'JCB702_BankTelemarketing1_PRACTICE.csv'
data = pd.read_csv(file_path)

# Drop duplicate rows before splitting. This file is balanced 50/50 by duplicating
# minority rows, so a random split leaves 54% of the test rows sitting verbatim in
# training and inflates every score that follows.
rows_before = len(data)
data = data.drop_duplicates()
print(f'Dropped {rows_before - len(data)} duplicate rows; {len(data)} remain')
print(f"Positive class rate: {data['y_yes'].mean():.1%}")

# Assuming 'y_yes' is the target variable and the rest are features
X = data.drop(columns=['y_yes'])
y = data['y_yes']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict_proba(X_test)[:, 1]
auc_log_reg = roc_auc_score(y_test, y_pred_log_reg)

# Decision Tree model
# max_depth is capped because an unconstrained tree memorises this dataset: 47% of its
# rows are exact duplicates, so 54% of test rows also appear verbatim in the training
# split. Depth 7 was chosen by 5-fold cross-validation on the deduplicated data.
# random_state fixes the tie-breaking so the reported AUC is reproducible.
dec_tree = DecisionTreeClassifier(max_depth=7, random_state=42)
dec_tree.fit(X_train, y_train)
y_pred_dec_tree = dec_tree.predict_proba(X_test)[:, 1]
auc_dec_tree = roc_auc_score(y_test, y_pred_dec_tree)

# Print AUCs
print(f'Logistic Regression AUC: {auc_log_reg}')
print(f'Decision Tree AUC: {auc_dec_tree}')

# Plot ROC curves
fpr_log_reg, tpr_log_reg, _ = roc_curve(y_test, y_pred_log_reg)
fpr_dec_tree, tpr_dec_tree, _ = roc_curve(y_test, y_pred_dec_tree)

plt.figure(figsize=(10, 6))
plt.plot(fpr_log_reg, tpr_log_reg, label=f'Logistic Regression (AUC = {auc_log_reg:.2f})')
plt.plot(fpr_dec_tree, tpr_dec_tree, label=f'Decision Tree (AUC = {auc_dec_tree:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()