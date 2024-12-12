# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = {
    'file_size_gb': [2.5, 1.0, 3.0, 4.5, 0.8, 1.2, 2.8, 3.5, 0.9, 4.0],
    'polygon_count': [1200000, 500000, 2000000, 3000000, 400000, 800000, 1500000, 2500000, 450000, 2800000],
    'texture_resolution': [4, 2, 8, 8, 2, 4, 6, 8, 2, 8],  # Resolution in K
    'frame_range': [250, 100, 300, 500, 50, 150, 350, 400, 75, 450],
    'render_engine': [1, 0, 1, 1, 0, 1, 0, 1, 0, 1],  # 1 = Cycles, 0 = Eevee
    'lighting_complexity': [5, 2, 8, 10, 1, 4, 6, 9, 2, 8],  # Arbitrary scale
    'render_time_hrs': [12, 1.5, 20, 35, 0.8, 5, 15, 30, 1.2, 25],  # Target variable
    'output_size_gb': [3.8, 0.8, 10.0, 20.0, 0.5, 1.2, 6.0, 15.0, 0.7, 12.5]  # Frame output size (target)
}

# Convert the dataset to a Pandas DataFrame
df = pd.DataFrame(data)

# Step 2: Split the dataset into features (X) and targets (y1 for render time, y2 for output size)
X = df[['file_size_gb', 'polygon_count', 'texture_resolution', 'frame_range', 'render_engine', 'lighting_complexity']]
y_time = df['render_time_hrs']
y_size = df['output_size_gb']

# Split the data into training and testing sets for both targets
X_train_time, X_test_time, y_train_time, y_test_time = train_test_split(X, y_time, test_size=0.2, random_state=42)
X_train_size, X_test_size, y_train_size, y_test_size = train_test_split(X, y_size, test_size=0.2, random_state=42)

# Step 3: Initialize Decision Tree Regressors
time_regressor = DecisionTreeRegressor(max_depth=5, random_state=42)
size_regressor = DecisionTreeRegressor(max_depth=5, random_state=42)

# Step 4: Train the models
time_regressor.fit(X_train_time, y_train_time)
size_regressor.fit(X_train_size, y_train_size)

# Step 5: Evaluate the models
y_pred_time = time_regressor.predict(X_test_time)
y_pred_size = size_regressor.predict(X_test_size)

# Calculate evaluation metrics for both models
mse_time = mean_squared_error(y_test_time, y_pred_time)
r2_time = r2_score(y_test_time, y_pred_time)
mse_size = mean_squared_error(y_test_size, y_pred_size)
r2_size = r2_score(y_test_size, y_pred_size)

print(f"Render Time Model - Mean Squared Error: {mse_time:.2f}, R-squared: {r2_time:.2f}")
print(f"Output Size Model - Mean Squared Error: {mse_size:.2f}, R-squared: {r2_size:.2f}")

# Display feature importances for both models
print("\nFeature Importances (Render Time Model):")
for feature, importance in zip(X.columns, time_regressor.feature_importances_):
    print(f"{feature}: {importance:.4f}")

print("\nFeature Importances (Output Size Model):")
for feature, importance in zip(X.columns, size_regressor.feature_importances_):
    print(f"{feature}: {importance:.4f}")

# Step 6: Use the models to make predictions on new data
new_project = [[3.0, 1500000, 8, 400, 1, 7]]
predicted_time = time_regressor.predict(new_project)
predicted_size = size_regressor.predict(new_project)

print(f"\nPredicted Render Time for new project: {predicted_time[0]:.2f} hours")
print(f"Predicted Memory Size for exported frames: {predicted_size[0]:.2f} GB")
