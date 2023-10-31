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
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)
class_names = ["lukostřelba", "baseball", "basketbal", "kulečník", "bmx", "bowling", "box", "jízda na býku", "roztleskávání", "curling", "šerm", "krasobruslení", "fotbal", "závody formule 1", "golf", "skok do výšky", "hokej", "dostihy", "hydroplánové závody", "judo", "motocyklové závody", "pole dance", "rugby", "skoky na lyžích", "snowboarding", "rychlobruslení", "surfování", "plavání", "stolní tenis", "tenis", "dráhové kolo", "volejbal", "vzpírání"]
'''def load_model():
    global model
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)

    model = tf.saved_model.load("../image_model/")

load_model()
'''
def UploadAction(event):
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
root.geometry('1500x800')

grey_frame = tk.Frame(root, bg="white", width=1050, height=800)
grey_frame.grid(row=0, column=0)

# Create a frame for the blue section
blue_frame = tk.Frame(root, bg="#D9FFFF", width=450, height=800)
blue_frame.grid(row=0, column=1)

image=Image.open('../components/bnz.png')
img=image.resize((296, 183))
photo=ImageTk.PhotoImage(img)

canvas = Canvas(bg="#D9FFFF", width=296, height=183, border=0, highlightthickness=0)
canvas.grid(row=0, column=1)
canvas.create_image(0, 0, anchor=NW, image=photo)
canvas.bind("<Button-1>", UploadAction)

root.mainloop()