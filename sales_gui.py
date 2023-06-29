import tkinter as tk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sales import sales_data

# Create a Tkinter window
window = tk.Tk()
window.title('Product Line Analysis')
window.geometry('800x600')

# Create a Figure object and add a subplot for the plot
figure = Figure(figsize=(6, 4), dpi=100)
subplot = figure.add_subplot(111)

# Group the data by product line and calculate total sales quantity
product_line_analysis = sales_data.groupby('Product line')['Quantity'].sum()

# Plot the total sales quantity by product line
product_line_analysis.plot(kind='bar', ax=subplot)
subplot.set_title('Total Sales Quantity by Product Line')
subplot.set_xlabel('Product Line')
subplot.set_ylabel('Total Sales Quantity')

# Create a FigureCanvasTkAgg object to display the plot in the Tkinter window
canvas = FigureCanvasTkAgg(figure, master=window)
canvas.draw()
canvas.get_tk_widget().pack()

# Start the Tkinter event loop
tk.mainloop()
