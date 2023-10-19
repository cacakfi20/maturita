import tkinter as tk
from tkinter import *
from tkinter import filedialog
import ctypes
from PIL import ImageTk, Image  

ctypes.windll.shcore.SetProcessDpiAwareness(True)

def UploadAction(event=None):
    filename = filedialog.askopenfilename()
    print('Selected:', filename)
    img = Image.open(filename)
    test = ImageTk.PhotoImage(img)
    label = tk.Label(image=test)
    label.grid(row=1, column=0)


root = tk.Tk()
root.title('Image Classification - Sport edition')
root.resizable(False, False)
root.configure(bg='black')
root.geometry('800x600')
root.grid_columnconfigure(0, weight=1)

button = tk.Button(root, text='Nahrát obrázek', command=UploadAction, width=30, height=4, bg='orange', fg='black', font=('Arial', 11, 'bold'))   
button.grid(row=0, column=0, pady=40)


root.mainloop()
