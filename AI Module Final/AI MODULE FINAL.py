import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from scipy import stats

# Print the current working directory
print("Current working directory:", os.getcwd())

# Load the new CSV file
file_path = r'D:\Codespace\PRACTICE FINAL\JCB702_BankTelemarketing2_PRACTICE.csv'
df = pd.read_csv(file_path)

# Display summary statistics
print("Summary Statistics:")
print(df.describe())

# Display data types
print("\nData Types:")
print(df.dtypes)

# Display missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Calculate the Z-score for each feature
z_scores = stats.zscore(df.select_dtypes(include=[float, int]))

# Remove outliers
df_cleaned = df[(abs(z_scores) < 3).all(axis=1)]

# Assuming 'y_yes' is the target variable and the rest are features
X = df_cleaned.drop('y_yes', axis=1)  # Features
y = df_cleaned['y_yes']  # Target variable

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree model
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

# Calculate and print feature importance scores
feature_importances = tree_model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\nFeature Importance Scores:")
print(importance_df)

# Plot feature importance scores
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance Scores')
plt.gca().invert_yaxis()
plt.savefig('feature_importance.png')
print("Feature importance plot saved as 'feature_importance.png'")