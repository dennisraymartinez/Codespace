import pandas as pd

# Load the CSV file
file_path = r'D:\Codespace\Compus\JCB703_COMPAS.csv'
df = pd.read_csv(file_path)

# Display all column names in a single row
print("Column Names:")
print(", ".join(df.columns))

# Create a new column 'predict_decile_5+' based on the condition
df['predict_decile_5+'] = df['prediction_decile_score.1'].apply(lambda x: 1 if x >= 5 else 0)

# Display the first few rows of the DataFrame to verify the new column
print("\nFirst few rows with the new column 'predict_decile_5+':")
print(df.head())

# Count the number of rows where 'predict_decile_5+' is equal to 1
count_predict_decile_5_plus = df[df['predict_decile_5+'] == 1].shape[0]
print(f"\nNumber of rows where 'predict_decile_5+' is equal to 1: {count_predict_decile_5_plus}")

# Calculate the proportion of 'predict_decile_5+' equal to 1 for each race
caucasian_df = df[df['race'] == 'Caucasian']
african_american_df = df[df['race'] == 'African-American']

caucasian_ppp = caucasian_df['predict_decile_5+'].mean()
african_american_ppp = african_american_df['predict_decile_5+'].mean()

print(f"\nProportion of 'predict_decile_5+' equal to 1 for Caucasian: {caucasian_ppp}")
print(f"Proportion of 'predict_decile_5+' equal to 1 for African-American: {african_american_ppp}")

# Calculate the disparate impact ratio
disparate_impact_ratio = african_american_ppp / caucasian_ppp
print(f"\nDisparate Impact Ratio (African-American / Caucasian): {disparate_impact_ratio}")

# Save the updated DataFrame back to a CSV file
updated_file_path = r'D:\Codespace\Compus\JCB703_COMPAS_updated.csv'
df.to_csv(updated_file_path, index=False)

print(f"Updated dataset has been saved to {updated_file_path}")

# Load the updated CSV file to verify the changes
updated_df = pd.read_csv(updated_file_path)

# Display the first few rows of the updated DataFrame
print("\nFirst few rows of the updated dataset:")
print(updated_df.head())