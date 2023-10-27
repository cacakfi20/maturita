import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import ctypes
from PIL import ImageTk, Image  
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

ctypes.windll.shcore.SetProcessDpiAwareness(True)
borderImageData = '''
    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
'''
class_names = ["lukostřelba", "baseball", "basketbal", "kulečník", "bmx", "bowling", "box", "jízda na býku", "roztleskávání", "curling", "šerm", "krasobruslení", "fotbal", "závody formule 1", "golf", "skok do výšky", "hokej", "dostihy", "hydroplánové závody", "judo", "motocyklové závody", "pole dance", "rugby", "skoky na lyžích", "snowboarding", "rychlobruslení", "surfování", "plavání", "stolní tenis", "tenis", "dráhové kolo", "volejbal", "vzpírání"]
'''def load_model():
    global model
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)

    model = tf.saved_model.load("../image_model/")

load_model()
'''
def UploadAction():
    filename = filedialog.askopenfilename()
    print('Selected:', filename)
    img = Image.open(filename)
    img = ImageTk.PhotoImage(img)  # Převedení obrázku na ImageTk.PhotoImage
    if hasattr(UploadAction, 'label'):
        UploadAction.label.destroy()
        
    UploadAction.label = tk.Label(image=img)
    UploadAction.label.image = img  # Uložení reference na obrázek
    UploadAction.label.grid(row=1, column=0)
    classify_image(filename)

def classify_image(img_url):
    img = image.load_img(img_url, target_size=(64, 64))
    X = image.img_to_array(img)
    X = np.expand_dims(X, axis=0)
    prediction = model(X)
    predicted_class_index = tf.argmax(prediction, axis=1).numpy()[0]
    predicted_class_label = class_names[predicted_class_index]

    if hasattr(classify_image, 'klasifikace'):
        classify_image.klasifikace.destroy()

    classify_image.klasifikace = tk.Label(root, text=f"Jedná se o {predicted_class_label} s pravděpodobností: {+100 * np.max(prediction)//1}%", fg='black', bg='white', font=('Arial', 11, 'bold'))
    classify_image.klasifikace.grid(row=2, column=0)
    print("Jedná se o", predicted_class_label, "s pravděpodobností:", 100 * np.max(prediction)//1, "%")

root = tk.Tk()
root.title('Image Classification - Sport edition')
root.resizable(True, True)
root.configure(bg='grey')
#root.attributes('-fullscreen',True)
root.geometry('1000x800')
root.grid_columnconfigure(0, weight=1)

#button = tk.Button(root, text='Nahrát obrázek', command=UploadAction, width=30, height=4, bg='#cc0066', fg='white', font=('Arial', 11, 'bold'))   
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)
img_button = PhotoImage(file='../components/bnz.png')
button = tk.Button(image=img_button, command=UploadAction, borderwidth=0)
button.grid(row=0, column=0, pady=40)

root.mainloop()