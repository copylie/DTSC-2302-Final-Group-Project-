import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('final_fixed_dataset.csv')

# Define features and target
X = data.drop(['home_price_2023', 'NPA'], axis=1)  # NPA is just an ID, not a predictor
y = data['home_price_2023']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:,.2f}')
print(f'RMSE: {np.sqrt(mse):,.2f}')  # easier to interpret — in same units as home price
print(f'R² Score: {r2:.4f}')

# Show feature importance (coefficients)
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values('Coefficient', ascending=False)

print('\nFeature Coefficients:')
print(coef_df.to_string(index=False))