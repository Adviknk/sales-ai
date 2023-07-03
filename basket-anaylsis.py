import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Read the CSV file
data = pd.read_csv('supermarket_sales_data.csv')

# Data preprocessing
data['Date'] = pd.to_datetime(data['Date'])
data['Time'] = pd.to_datetime(data['Time']).dt.hour

# Predicting Customer Ratings

# Select relevant features for prediction
rating_features = ['Customer type', 'Gender', 'Unit price', 'Quantity', 'Tax 5%',
                   'Total', 'Time', 'Payment', 'cogs', 'gross margin percentage', 'gross income']

# Split the data into features and target variable
X = data[rating_features]
y = data['Rating']

categorical_features = ['Branch', 'City',
                        'Customer type', 'Gender', 'Product line', 'Payment']
preprocessor = ColumnTransformer(transformers=[(
    'cat', OneHotEncoder(), categorical_features)], remainder='passthrough')
X_encoded = preprocessor.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train a random forest regression model
rating_model = RandomForestRegressor(n_estimators=100, random_state=42)
rating_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rating_model.predict(X_test)

# Calculate the root mean squared error (RMSE)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Root Mean Squared Error (RMSE) for Rating Prediction: {rmse}")

# Market Basket Analysis

# Perform one-hot encoding for categorical features
market_basket_data = pd.get_dummies(
    data[['Invoice id', 'Product line']], columns=['Product line'])
market_basket_data = market_basket_data.groupby(
    'Invoice id').sum().applymap(lambda x: 1 if x >= 1 else 0)

# Perform market basket analysis using Apriori algorithm
frequent_itemsets = apriori(
    market_basket_data, min_support=0.05, use_colnames=True)
association_rules_df = association_rules(
    frequent_itemsets, metric="lift", min_threshold=1)

# Print the association rules
print("Association Rules:")
print(association_rules_df)

# Print frequent itemsets and their support
print("\nFrequent Itemsets:")
print(frequent_itemsets)
