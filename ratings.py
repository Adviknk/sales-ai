import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data = pd.read_csv('supermarket_sales_data.csv')

data['Date'] = pd.to_datetime(data['Date'])
data['Time'] = pd.to_datetime(data['Time']).dt.hour
data['Customer type'] = LabelEncoder().fit_transform(data['Customer type'])
data['Gender'] = LabelEncoder().fit_transform(data['Gender'])
data['Payment'] = LabelEncoder().fit_transform(data['Payment'])

categorical_cols = ['Branch', 'City', 'Product line']
data = pd.get_dummies(data, columns=categorical_cols)

features = ['Customer type', 'Gender', 'Unit price', 'Quantity', 'Tax 5%',
            'Total', 'Time', 'Payment', 'cogs', 'gross margin percentage', 'gross income']

X_train, X_test, y_train, y_test = train_test_split(
    data[features], data['Rating'], test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Root Mean Squared Error (RMSE): {rmse}")

plt.scatter(y_test, y_pred)
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Actual vs. Predicted Ratings')
plt.show()
