import pandas as pd
import matplotlib.pyplot as plt

# Load the sales data into a DataFrame
sales_data = pd.read_csv('supermarket_sales_data.csv')

# Group the data by product line and calculate total sales quantity and revenue
product_line_analysis = sales_data.groupby(
    'Product line').agg({'Quantity': 'sum', 'Total': 'sum'})

# Sort the analysis results by total sales quantity in descending order
product_line_analysis = product_line_analysis.sort_values(
    'Quantity', ascending=False)

# Plot the total sales quantity by product line
plt.figure(figsize=(10, 6))
product_line_analysis['Quantity'].plot(kind='bar', color='blue')
plt.title('Total Sales Quantity by Product Line')
plt.xlabel('Product Line')
plt.ylabel('Total Sales Quantity')
plt.xticks(rotation=45)
plt.show()

# Sort the analysis results by total revenue in descending order
product_line_analysis = product_line_analysis.sort_values(
    'Total', ascending=False)

# Plot the total revenue by product line
plt.figure(figsize=(10, 6))
product_line_analysis['Total'].plot(kind='bar', color='green')
plt.title('Total Revenue by Product Line')
plt.xlabel('Product Line')
plt.ylabel('Total Revenue ($)')
plt.xticks(rotation=45)
plt.show()
