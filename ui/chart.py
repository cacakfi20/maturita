import matplotlib.figure
import matplotlib.patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

fig = matplotlib.figure.Figure(figsize=(5,5))
ax = fig.add_subplot(111)
ax.pie([80,20], colors=["green","white"], startangle=90) 
ax.legend(["50","50"])

circle=matplotlib.patches.Circle((0,0), 0.8, color='white')
ax.add_artist(circle)

window= tk.Tk()
canvas = FigureCanvasTkAgg(fig, master=window)
canvas.get_tk_widget().pack()
canvas.draw()
window.mainloop()