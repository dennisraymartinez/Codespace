import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# CSV data string
csv_data = """Manufacturer,Product_price,Price_Low,Price_Mid,Price_High,Product_ratings,One_Star,Two_Star,Three_Star,Four_Star,Product_name,Noise_Cancelling,Product_url,Product_price_url,Product_price_clean
Soundcore,49.0,0,1,0,4.5,1,0,0,0,"Soundcore Anker Life Q20 Hybrid Active Noise Cancelling Headphones, Wireless Over Ear Bluetooth Headphones, 60H Playtime, Hi-Res Audio, Deep Bass, Memory Foam Ear Cups, for Travel, Home Office",0,https://www.amazon.com/Soundcore-Cancelling-Headphones-Wireless-Bluetooth/dp/B07NM3RSRQ/ref=sr_1_11,https://www.amazon.com/Soundcore-Cancelling-Headphones-Wireless-Bluetooth/dp/B07NM3RSRQ/ref=sr_1_11,49.0
Uliptz,17.0,1,0,0,4.5,0,0,0,0,"Uliptz Wireless Bluetooth Headphones, 65H Playtime, 6 EQ Sound Modes, HiFi Stereo Over Ear Headphones with Microphone, Foldable Lightweight Bluetooth 5.3 Headphones for Travel/Office/Cellphone/PC",0,https://www.amazon.com/Uliptz-Bluetooth-Headphones-Microphone-Lightweight/dp/B09NNBBY8F/ref=sr_1_12,https://www.amazon.com/Uliptz-Bluetooth-Headphones-Microphone-Lightweight/dp/B09NNBBY8F/ref=sr_1_12,17.0
JBL,59.0,0,0,1,4.5,0,1,0,0,"JBL Tune 670NC - Adaptive Noise Cancelling with Smart Ambient Wireless On-Ear Headphones, Up to 70H Battery Life with Speed Charge, Lightweight, Comfortable and Foldable Design (Black)",0,https://www.amazon.com/JBL-TUNE-670NC-Ear-Lightweight/dp/B0CT9MB4WR/ref=sr_1_15,https://www.amazon.com/JBL-TUNE-670NC-Ear-Lightweight/dp/B0CT9MB4WR/ref=sr_1_15,59.0
Beats,99.0,0,0,1,4.5,0,1,0,0,"Beats Solo 4 - Wireless Bluetooth On-Ear Headphones, Apple & Android Compatible, Up to 50 Hours of Battery Life - Matte Black",0,https://www.amazon.com/Beats-Solo-Wireless-Headphones-Matte/dp/B0CZPLV566/ref=sr_1_16,https://www.amazon.com/Beats-Solo-Wireless-Headphones-Matte/dp/B0CZPLV566/ref=sr_1_16,99.0
Bluetooth,19.0,1,0,0,4.4,0,0,0,0,"KVIDIO [Updated Bluetooth Headphones Over Ear, 65 Hours Playtime Wireless Headphones with Microphone,Foldable Lightweight Headset with Deep Bass,HiFi Stereo Sound for Travel Work Cellphone",0,https://www.amazon.com/Bluetooth-Headphones-KVIDIO-Microphone-Lightweight/dp/B09BF64J55/ref=sr_1_21,https://www.amazon.com/Bluetooth-Headphones-KVIDIO-Microphone-Lightweight/dp/B09BF64J55/ref=sr_1_21,19.0
Active,39.0,0,1,0,4.3,0,0,0,0,"MOVSSOU E7 Active Noise Cancelling Headphones Bluetooth Headphones Wireless Headphones Over Ear with Microphone Deep Bass, Comfortable Protein Earpads, 30 Hours Playtime for Travel/Work, Black",0,https://www.amazon.com/Active-Noise-Cancelling-Headphones-Comfortable/dp/B095BV8R27/ref=sr_1_22,https://www.amazon.com/Active-Noise-Cancelling-Headphones-Comfortable/dp/B095BV8R27/ref=sr_1_22,39.0
Soundcore,51.0,0,1,0,4.5,0,0,0,0,"Soundcore by Anker Life Q30 Hybrid Active Noise Cancelling Headphones, Wireless Over Ear Bluetooth Headphones, 40H Playtime, Hi-Res Audio, Deep Bass, Memory Foam Ear Cups, for Travel, Home Office",0,https://www.amazon.com/Soundcore-Cancelling-Headphones-Wireless-Bluetooth/dp/B08HMWZBXC/ref=sr_1_23,https://www.amazon.com/Soundcore-Cancelling-Headphones-Wireless-Bluetooth/dp/B08HMWZBXC/ref=sr_1_23,51.0
"""

# Load the CSV data into a DataFrame
df = pd.read_csv(StringIO(csv_data))

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

# Identify outliers
outliers = df[(abs(z_scores) >= 3).any(axis=1)]
print("\nOutliers:")
print(outliers)

# Remove outliers
df_cleaned = df[(abs(z_scores) < 3).all(axis=1)]

# Check if the cleaned dataset is empty
if df_cleaned.empty:
    print("The cleaned dataset is empty after removing outliers.")
else:
    # Assuming 'Product_price' is the target variable and the rest are features
    X = df_cleaned.drop(['Product_price', 'Manufacturer', 'Product_name', 'Product_url', 'Product_price_url'], axis=1)  # Features
    y = df_cleaned['Product_price']  # Target variable

    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Decision Tree Regressor model
    tree_model = DecisionTreeRegressor(random_state=42)
    tree_model.fit(X_train, y_train)

    # Make predictions
    y_pred = tree_model.predict(X_test)

    # Calculate R-squared value
    r2 = r2_score(y_test, y_pred)
    print(f'R-squared: {r2}')