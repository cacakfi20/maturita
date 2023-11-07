import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import os
import ctypes
from PIL import ImageTk, Image
import tensorflow as tf
import numpy as np
import matplotlib.figure
import matplotlib.patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorflow.keras.preprocessing import image

class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Image Classification - Sport edition')
        self.root.resizable(False, False)
        self.root.configure(bg='grey')
        self.create_image_ui()

    def create_image_ui(self):
        self.class_names = ["lukostřelba", "baseball", "basketbal", "kulečník", "bmx", "bowling", "box", "jízda na býku", "roztleskávání", "curling", "šerm", "krasobruslení", "americký fotbal", "závody formule 1", "golf", "skok do výšky", "hokej", "dostihy", "závody hydroplánů", "judo", "motocyklové závody", "pole dance", "rugby", "skoky na lyžích", "snowboarding", "rychlobruslení", "surfování", "plavání", "stolní tenis", "tenis", "dráhové kolo", "volejbal", "vzpírání"]
        
        self.grey_frame = tk.Frame(self.root, bg="white", width=1050, height=800)
        self.grey_frame.grid(row=0, column=0)

        self.blue_frame = tk.Frame(self.root, bg="#D9FFFF", width=450, height=800)
        self.blue_frame.grid(row=0, column=1)

        script_directory = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_directory)

        self.create_toggle_menu()

        imagebtn=Image.open('../components/button.png')
        imgbtn=imagebtn.resize((296, 183))
        self.photo=ImageTk.PhotoImage(imgbtn)

        self.button_canvas = tk.Canvas(bg="#D9FFFF", width=296, height=183, border=0, highlightthickness=0)
        self.button_canvas.grid(sticky=tk.N, row=0, column=1, pady=(60,0))
        self.button_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.button_canvas.bind("<Button-1>", self.upload_action)
        self.button_canvas.bind("<Enter>", self.change_cursor)
        self.button_canvas.bind("<Leave>", self.restore_cursor)

        imageLogo=Image.open('../components/logo.png')
        imgLogo=imageLogo.resize((169, 33))
        self.photoLogo=ImageTk.PhotoImage(imgLogo)

        self.logo_canvas = tk.Canvas(bg="white", width=169, height=33, border=0, highlightthickness=0)
        self.logo_canvas.grid(sticky=tk.SW, row=0, column=0, pady=(0,50), padx=(100,0))
        self.logo_canvas.create_image(0, 0, anchor=tk.NW, image=self.photoLogo)

        self.name_label = tk.Label(text="© 2023 Filip Cacák", bg="white", fg="#d6d6d6", font=('Arial', 16, 'bold'))
        self.name_label.grid(sticky=tk.SW, row=0, column=0, pady=(0,50), padx=(290,0))

    def upload_action(self, event):
        filename = filedialog.askopenfilename()
        sport_img = Image.open(filename)
        max_width = 850
        max_height = 600
        width, height = sport_img.size
        width_scale = max_width / width
        height_scale = max_height / height
        scale_factor = min(width_scale, height_scale)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        sport_img = sport_img.resize((new_width, new_height))
        self.sport_photo = ImageTk.PhotoImage(sport_img)

        if hasattr(self, 'label'):
            self.label.destroy()

        self.label = tk.Label(image=self.sport_photo)
        self.label.image = self.photo
        self.label.grid(row=0, column=0)
        self.classify_image(filename)

    def classify_image(self, img_url):
        img = image.load_img(img_url, target_size=(64, 64))
        X = image.img_to_array(img)
        X = np.expand_dims(X, axis=0)
        prediction = self.model(X)
        predicted_class_index = tf.argmax(prediction, axis=1).numpy()[0]
        predicted_class_label = self.class_names[predicted_class_index]

        if hasattr(self, 'pred_label'):
            self.pred_label.destroy()
        if hasattr(self, 'desc_label'):
            self.desc_label.destroy()
        
        script_directory = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_directory)

        fig = matplotlib.figure.Figure(figsize=(1.5,1.5))
        fig.patch.set_facecolor('#D9FFFF')
        ax = fig.add_subplot(111)
        if 100 * np.max(prediction)//1 <= 30:
            ax.pie([100 * np.max(prediction)//1,100-100 * np.max(prediction)//1], colors=["#ff6666","#D9FFFF"], startangle=90)   
        if 100 * np.max(prediction)//1 > 30:
            ax.pie([100 * np.max(prediction)//1,100-100 * np.max(prediction)//1], colors=["#e68541","#D9FFFF"], startangle=90)    
        if 100 * np.max(prediction)//1 >= 70:
            ax.pie([100 * np.max(prediction)//1,100-100 * np.max(prediction)//1], colors=["#53af32","#D9FFFF"], startangle=90) 
        circle=matplotlib.patches.Circle((0,0), 0.8, color='#D9FFFF')
        ax.add_artist(circle)
        self.canvasChart = FigureCanvasTkAgg(fig, master=root,)
        self.canvasChart.get_tk_widget().grid(sticky=tk.NE ,row=0, column=1, pady=(360,0), padx=(0, 27))
        self.canvasChart.draw()

        pred_label=Image.open('../components/prediction.png')
        pred_img=pred_label.resize((296, 90))
        self.pred_photo=ImageTk.PhotoImage(pred_img)

        self.pred_canvas = tk.Canvas(bg="#D9FFFF", width=296, height=90, border=0, highlightthickness=0)
        self.pred_canvas.grid(sticky=tk.NW, row=0, column=1, pady=(300,0), padx=(75,0))
        self.pred_canvas.create_image(0, 0, anchor=tk.NW, image=self.pred_photo)

        self.pred_label = tk.Label(bg="#D9FFFF", text=f"{int(100 * np.max(prediction)//1)} %", font=('Arial', 18, 'bold'))
        self.pred_label.grid(sticky=tk.NE ,row=0, column=1, pady=(432,0), padx=(0, 80))

        desc_label=Image.open('../components/description.png')
        desc_img=desc_label.resize((115, 33))
        self.desc_photo=ImageTk.PhotoImage(desc_img)

        self.desc_canvas = tk.Canvas(bg="#D9FFFF", width=115, height=33, border=0, highlightthickness=0)
        self.desc_canvas.grid(sticky=tk.NW, row=0, column=1, pady=(400,0), padx=(75,0))
        self.desc_canvas.create_image(0, 0, anchor=tk.NW, image=self.desc_photo)

        if len(predicted_class_label.split()) == 2:
            predicted_class_label = predicted_class_label.split()[0] + "\n" + predicted_class_label.split()[1]
        if len(predicted_class_label.split()) == 3:
            predicted_class_label = predicted_class_label.split()[0] + "\n" + predicted_class_label.split()[1] + " " + predicted_class_label.split()[2]
        if len(predicted_class_label) > 10:
            self.desc_label = tk.Label(bg="#D9FFFF", text=predicted_class_label.upper(), font=('Arial', 12, 'bold'))
        else:
            self.desc_label = tk.Label(bg="#D9FFFF", text=predicted_class_label.upper(), font=('Arial', 15, 'bold'))

        self.desc_label.grid(sticky=tk.NW ,row=0, column=1, pady=(450,0), padx=(75, 0))

    def change_cursor(self, event):
        self.button_canvas.config(cursor="hand2")

    def restore_cursor(self, event):
        self.button_canvas.config(cursor="")

    def load_model(self):
        script_directory = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_directory)
        self.model = tf.saved_model.load("../image_model/")

    def create_toggle_menu(self):
        menu_image=Image.open('../components/toggle_menu.png')
        menu_img=menu_image.resize((50, 50))
        self.menu_photo=ImageTk.PhotoImage(menu_img)

        self.menu_btn_canvas = tk.Canvas(bg="white", width=50, height=50, border=0, highlightthickness=0)
        self.menu_btn_canvas.grid(sticky=tk.NW, row=0, column=0, pady=(20,0), padx=(20,0))
        self.menu_btn_canvas.create_image(0, 0, anchor=tk.NW, image=self.menu_photo)
        self.menu_btn_canvas.bind("<Button-1>", self.toggle_menu)
        self.menu_btn_canvas.config(cursor="hand2")

    def toggle_menu(self, event):
        items = ['Obrázková klasifikace', 'Textová klasifikace', 'Nápověda', 'O autorovi', 'O aplikaci']
        self.opened_menu = tk.Frame(self.root, bg="#f5f5f5", width=250, height=800)
        self.opened_menu.grid(row=0, column=0, sticky=tk.W)

        self.opened_menu.grid_rowconfigure(0, weight=1)  # Keep the row at a fixed height

        close_menu_image = Image.open('../components/close_menu.png')
        close_menu_img = close_menu_image.resize((50, 50))
        self.close_menu_photo = ImageTk.PhotoImage(close_menu_img)

        self.close_menu_btn_canvas = tk.Canvas(bg="#f5f5f5", width=50, height=50, border=0, highlightthickness=0)
        self.close_menu_btn_canvas.grid(sticky=tk.NW, row=0, column=0, pady=(20, 0), padx=(20, 0))
        self.close_menu_btn_canvas.create_image(0, 0, anchor=tk.NW, image=self.close_menu_photo)
        self.close_menu_btn_canvas.bind("<Button-1>", self.close_menu)
        self.close_menu_btn_canvas.config(cursor="hand2")

        self.menu_item_labels = []

        for i, item_text in enumerate(items):
            menu_item = tk.Label(self.root, text=item_text, bg='#f5f5f5', font=('Arial', 11, 'bold'))
            if i == 0:
                menu_item.grid(row=0, column=0, sticky=tk.NW, pady=(150, 20), padx=(20, 0))
            elif i == 1:
                menu_item.grid(row=0, column=0, sticky=tk.NW, pady=(190, 0), padx=(20, 0))
            else:  
                menu_item.grid(row=0, column=0, sticky=tk.SW, pady=(0, 40*i), padx=(20, 0))
            menu_item.config(cursor="hand2")
            menu_item.bind("<Button-1>", lambda event, ui_name=item_text: self.switch_ui(ui_name))

            self.menu_item_labels.append(menu_item)

    
    def close_menu(self, event):
        self.opened_menu.destroy()
        self.close_menu_btn_canvas.destroy()
        for menu_item in self.menu_item_labels:
            menu_item.destroy()
        self.menu_item_labels = []
    
    def switch_ui(self, ui_name):
        for child in root.winfo_children():
            child.destroy()

        if ui_name == 'Obrázková klasifikace':
            self.create_image_ui()
        elif ui_name == 'Textová klasifikace':
            self.create_text_ui()
        elif ui_name == 'O aplikaci':
            self.create_about_ui()
        elif ui_name == 'O autorovi':
            self.create_me_ui()
        elif ui_name == 'Nápověda':
            self.create_help_ui()

    def create_text_ui(self):
        self.grey_frame = tk.Frame(self.root, bg="white", width=1050, height=800)
        self.grey_frame.grid(row=0, column=0)

        self.blue_frame = tk.Frame(self.root, bg="#D9FFFF", width=450, height=800)
        self.blue_frame.grid(row=0, column=1)

        script_directory = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_directory)

        self.create_toggle_menu()

        lbl = tk.Label(text="Textová klasifikace")
        lbl.grid(row=0, column=0)
    
    def create_about_ui(self):
        self.grey_frame = tk.Frame(self.root, bg="white", width=1050, height=800)
        self.grey_frame.grid(row=0, column=0)

        self.blue_frame = tk.Frame(self.root, bg="#D9FFFF", width=450, height=800)
        self.blue_frame.grid(row=0, column=1)

        script_directory = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_directory)

        self.create_toggle_menu()

        lbl = tk.Label(text="O aplikaci", font=('Arial', 20, 'bold'), bg="white", fg="#53af32")
        lbl.grid(row=0, column=0, sticky=tk.N, pady=(20,0))

        text = tk.Text()
        text.grid(row=0, column=0, sticky=tk.N, pady=(100,0), padx=(50,0))
        text.insert(tk.END, "Sport Classifier Pro je inovativní aplikace vytvořená jako součást maturitního projektu, která využívá pokročilé strojové učení pro klasifikaci sportů na základě obrázků a textových popisů. Naše aplikace vám umožní snadno a rychle zjistit, o jaký sport se jedná, a to jak na základě fotografií, tak i na základě textových popisů")

    def create_me_ui(self):
        self.grey_frame = tk.Frame(self.root, bg="white", width=1050, height=800)
        self.grey_frame.grid(row=0, column=0)

        self.blue_frame = tk.Frame(self.root, bg="#D9FFFF", width=450, height=800)
        self.blue_frame.grid(row=0, column=1)

        script_directory = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_directory)

        self.create_toggle_menu()

        lbl = tk.Label(text="O autorovi")
        lbl.grid(row=0, column=0)

    def create_help_ui(self):
        self.grey_frame = tk.Frame(self.root, bg="white", width=1050, height=800)
        self.grey_frame.grid(row=0, column=0)

        self.blue_frame = tk.Frame(self.root, bg="#D9FFFF", width=450, height=800)
        self.blue_frame.grid(row=0, column=1)

        script_directory = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_directory)

        self.create_toggle_menu()

        lbl = tk.Label(text="Nápověda")
        lbl.grid(row=0, column=0)

if __name__ == "__main__":
    ctypes.windll.shcore.SetProcessDpiAwareness(True)
    root = tk.Tk()
    app = ImageClassifierApp(root)
    #app.load_model()
    root.mainloop()
