import pandas as pd
import matplotlib.pyplot as plt

sales_data = pd.read_csv('supermarket_sales_data.csv')

product_line_analysis = sales_data.groupby(
    'Product line').agg({'Quantity': 'sum', 'Total': 'sum'})

product_line_analysis = product_line_analysis.sort_values(
    'Quantity', ascending=False)

plt.figure(figsize=(10, 6))
product_line_analysis['Quantity'].plot(kind='bar', color='blue')
plt.title('Total Sales Quantity by Product Line')
plt.xlabel('Product Line')
plt.ylabel('Total Sales Quantity')
plt.xticks(rotation=45)
plt.show()

product_line_analysis = product_line_analysis.sort_values(
    'Total', ascending=False)

plt.figure(figsize=(10, 6))
product_line_analysis['Total'].plot(kind='bar', color='green')
plt.title('Total Revenue by Product Line')
plt.xlabel('Product Line')
plt.ylabel('Total Revenue ($)')
plt.xticks(rotation=45)
plt.show()
