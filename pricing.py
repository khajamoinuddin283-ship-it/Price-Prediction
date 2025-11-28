import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def get_inflation_rate_india():
    url = "https://api.api-ninjas.com/v1/inflation?country=India"  
    headers = {
        'X-Api-Key': 'Muy9b2OrYI3m0kl3QmXJL0l4LEWgjlm4RxxCbqRI' 
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        if response.status_code == 200:
            inflation_data = response.json()
            if inflation_data and isinstance(inflation_data, list) and 'inflation_rate' in inflation_data[0]:
                inflation_rate = inflation_data[0]['inflation_rate']
                return inflation_rate
        return 1.5  # Default value if API request fails
    except requests.exceptions.RequestException as e:
        return 1.5  # Default inflation rate

def calculate_selling_price_with_profit_range(total_cost, min_profit=10, max_profit=30):
    selling_price_min = total_cost * (1 + min_profit / 100)
    selling_price_max = total_cost * (1 + max_profit / 100)
    return selling_price_min, selling_price_max

def predict_product_price_with_range_and_profit(production_cost, labour_cost, 
                                                raw_material_cost, rent, advertising,transportation_cost_percentage, packet_size, gst_percentage):
    inflation = get_inflation_rate_india()

    # Calculate total cost
    total_cost = (
        production_cost + labour_cost + raw_material_cost + rent +
        advertising + transportation_cost_percentage +(production_cost * gst_percentage)
    )

    # Predict the selling price using the model
    input_data = pd.DataFrame([{
        'production_cost': production_cost,
        'labour_cost': labour_cost,
        'raw_material_cost': raw_material_cost,
        'rent': rent,
        'advertising': advertising,
        'transportation_cost_percentage': transportation_cost_percentage,
        'packet_size': packet_size,
        'gst_percentage': gst_percentage
        
    }])

    predicted_price = best_model.predict(input_data)[0]
    adjusted_price = predicted_price * inflation  # Adjust price with inflation

    # Calculate the profit range
    selling_price_min, selling_price_max = calculate_selling_price_with_profit_range(total_cost, min_profit=10, max_profit=30)
    adjusted_price = max(selling_price_min, min(selling_price_max, adjusted_price))

    residuals = y_test - best_model.predict(X_test)
    residual_std = np.std(residuals)
    price_lower_bound = adjusted_price - residual_std
    price_upper_bound = adjusted_price + residual_std

    return adjusted_price, price_lower_bound, price_upper_bound
class ChipClassifier:
    def __init__(self):
        
        self.clusters = {
            "Regular Pack": 30,
            "Medium Pack": 60,
            "Family Pack": 100
        }

    def classify(self, grams):
        nearest_label = None
        min_distance = float('inf')
        for label, centroid in self.clusters.items():
            distance = abs(grams - centroid)
            if distance < min_distance:
                min_distance = distance
                nearest_label = label
        return nearest_label



file_path = r"C:\Users\91966\ai_pricing_tool\product_data2.csv" 
data = pd.read_csv(file_path)


if 'packet_size' not in data.columns:
    np.random.seed(42)
    data['packet_size'] = np.random.choice([50, 100, 150, 200, 250], size=len(data))


X = data[['production_cost',  'labour_cost',
          'raw_material_cost', 'rent', 'advertising','transportation_cost_percentage',  'packet_size','gst_percentage']]
y = data['selling_price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


pipeline = Pipeline([('scaler', StandardScaler()), 
                     ('model', GradientBoostingRegressor(random_state=42))])

param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__learning_rate': [0.01, 0.1, 0.2],
    'model__max_depth': [3, 5, 7]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_


y_pred = best_model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
test_r2 = r2_score(y_test, y_pred)

print("Best Model Parameters:", grid_search.best_params_)
print(f"Test RMSE: {test_rmse}")
print(f"Test R² Score: {test_r2}")
