import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import os, sys
import ctypes
from PIL import ImageTk, Image
import tensorflow as tf
import numpy as np
import matplotlib.figure
import matplotlib.patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from configparser import ConfigParser
from tensorflow.keras.preprocessing import image
import pandas as pd

class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Image Classification - Sport edition')
        self.root.resizable(False, False)
        self.root.configure(bg='grey')
        self.create_image_ui()

    def get_settings(self):
        script_directory = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_directory)
        parser = ConfigParser()
        parser.read('./config.ini')

        self.primary_background = parser.get('options', 'primary_background')
        self.primary_foreground = parser.get('options', 'primary_foreground')
        self.secondary_background = parser.get('options', 'secondary_background')
        self.secondary_foreground = parser.get('options', 'secondary_foreground')
        self.sidebar_background = parser.get('options', 'sidebar_background')
        self.sidebar_foreground = parser.get('options', 'sidebar_foreground')

        self.button_image = parser.get('options', 'button_image')
        self.logo_image = parser.get('options', 'logo_image')
        self.close_menu_image = parser.get('options', 'close_menu_image')
        self.description_image = parser.get('options', 'description_image') 
        self.prediction_image = parser.get('options', 'prediction_image')
        self.toggle_menu_image = parser.get('options', 'toggle_menu_image')

        self.sport_description_file_url = parser.get('options', 'image_description_file')
        self.latest_image = parser.get('options', 'latest_image')

        self.language = parser.get('options', 'language')

    def create_image_ui(self):
        self.get_settings()
        self.class_names = ["lukostřelba", "baseball", "basketbal", "kulečník", "bmx", "bowling", "box", "jízda na býku", "roztleskávání", "curling", "šerm", "krasobruslení", "americký fotbal", "závody formule 1", "golf", "skok do výšky", "hokej", "dostihy", "závody hydroplánů", "judo", "motocyklové závody", "pole dance", "rugby", "skoky na lyžích", "snowboarding", "rychlobruslení", "surfování", "plavání", "stolní tenis", "tenis", "dráhové kolo", "volejbal", "vzpírání"]
        
        self.grey_frame = tk.Frame(self.root, bg=self.primary_background, width=1050, height=800)
        self.grey_frame.grid(row=0, column=0)

        self.blue_frame = tk.Frame(self.root, bg=self.secondary_background, width=450, height=800)
        self.blue_frame.grid(row=0, column=1)

        script_directory = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_directory)

        self.create_toggle_menu()
        try:
            sport_img = Image.open(self.latest_image)
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

            self.label = tk.Label(image=self.sport_photo)
            self.label.image = self.sport_photo
            self.label.grid(row=0, column=0)
            if hasattr(self, 'setting_image_loaded'):
                self.classify_image(self.latest_image)
            self.setting_image_loaded = True
        except:
            pass

        imagebtn=Image.open(self.button_image)
        imgbtn=imagebtn.resize((296, 183))
        self.photo=ImageTk.PhotoImage(imgbtn)

        self.button_canvas = tk.Canvas(bg=self.secondary_background, width=296, height=183, border=0, highlightthickness=0)
        self.button_canvas.grid(sticky=tk.N, row=0, column=1, pady=(60,0))
        self.button_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.button_canvas.bind("<Button-1>", self.upload_action)
        self.button_canvas.bind("<Enter>", self.change_cursor)
        self.button_canvas.bind("<Leave>", self.restore_cursor)

        imageLogo=Image.open(self.logo_image)
        imgLogo=imageLogo.resize((169, 33))
        self.photoLogo=ImageTk.PhotoImage(imgLogo)

        self.logo_canvas = tk.Canvas(bg=self.primary_background, width=169, height=33, border=0, highlightthickness=0)
        self.logo_canvas.grid(sticky=tk.SW, row=0, column=0, pady=(0,50), padx=(100,0))
        self.logo_canvas.create_image(0, 0, anchor=tk.NW, image=self.photoLogo)

        self.name_label = tk.Label(text="© 2023 Filip Cacák", bg=self.primary_background, fg="#d6d6d6", font=('Arial', 16, 'bold'))
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
        parser = ConfigParser()
        parser.read('./config.ini')
        parser['options']['latest_image'] = filename
        with open('./config.ini', 'w') as configfile:
            parser.write(configfile)
        self.get_settings()

    def classify_image(self, img_url):

        if self.language == 'cz':
            self.class_names = ["lukostřelba", "baseball", "basketbal", "kulečník", "bmx", "bowling", "box", "jízda na býku", "roztleskávání", "curling", "šerm", "krasobruslení", "americký fotbal", "závody formule 1", "golf", "skok do výšky", "hokej", "dostihy", "závody hydroplánů", "judo", "motocyklové závody", "pole dance", "rugby", "skoky na lyžích", "snowboarding", "rychlobruslení", "surfování", "plavání", "stolní tenis", "tenis", "dráhové kolo", "volejbal", "vzpírání"]
        else:
            self.class_names = ["archery", "baseball", "basketball", "billiards", "bmx", "bowling", "boxing", "bull riding", "cheerleading", "curling", "fencing", "figure skating", "football", "formula 1", "golf", "high jump", "hockey", "horse racing", "hydroplane racing", "judo", "motorcycle racing", "pole dance", "rugby", "ski jumping", "snowboarding", "speed skating", "surfing", "swimming", "table tennis", "tennis", "track cycling", "volleyball", "weightlifting"]

        img = image.load_img(img_url, target_size=(64, 64))
        X = image.img_to_array(img)
        X = np.expand_dims(X, axis=0)
        prediction = self.model(X)
        predicted_class_index = tf.argmax(prediction, axis=1).numpy()[0]
        predicted_class_label = self.class_names[predicted_class_index]

        top = tf.argsort(prediction, axis=-1, direction='DESCENDING')
        for i in top.numpy()[0][:3]:
            probability = prediction[0][i].numpy()
            print(self.class_names[i], probability*100)

        if hasattr(self, 'pred_label'):
            self.pred_label.destroy()   
        if hasattr(self, 'desc_label'):
            self.desc_label.destroy()
        if hasattr(self, 'sport_desc_lbl'):
            self.sport_desc_lbl.destroy()
        
        script_directory = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_directory)

        self.pieCanvas = tk.Canvas(bg=self.secondary_background, width=296, height=296, border=0, highlightthickness=0)
        self.pieCanvas.grid(sticky=tk.NE, row=0, column=1, pady=(360,0), padx=(0, 27))

        fig = matplotlib.figure.Figure(figsize=(1.5,1.5))
        fig.patch.set_facecolor(self.secondary_background)
        ax = fig.add_subplot(111)
        if 100 * np.max(prediction)//1 <= 30:
            ax.pie([100 * np.max(prediction)//1,100-100 * np.max(prediction)//1], colors=["#ff6666",self.secondary_background], startangle=90)   
        if 100 * np.max(prediction)//1 > 30:
            ax.pie([100 * np.max(prediction)//1,100-100 * np.max(prediction)//1], colors=["#e68541",self.secondary_background], startangle=90)    
        if 100 * np.max(prediction)//1 >= 70:
            ax.pie([100 * np.max(prediction)//1,100-100 * np.max(prediction)//1], colors=["#53af32",self.secondary_background], startangle=90) 
        circle=matplotlib.patches.Circle((0,0), 0.8, color=self.secondary_background)
        ax.add_artist(circle)
        self.canvasChart = FigureCanvasTkAgg(fig, master=self.pieCanvas,)
        self.canvasChart.get_tk_widget().grid(row=0, column=0, pady=(0,0), padx=(0, 0))
        self.canvasChart.draw()

        pred_label=Image.open(self.prediction_image)
        pred_img=pred_label.resize((296, 90))
        self.pred_photo=ImageTk.PhotoImage(pred_img)

        self.pred_canvas = tk.Canvas(bg=self.secondary_background, width=296, height=90, border=0, highlightthickness=0)
        self.pred_canvas.grid(sticky=tk.NW, row=0, column=1, pady=(300,0), padx=(75,0))
        self.pred_canvas.create_image(0, 0, anchor=tk.NW, image=self.pred_photo)

        self.pred_label = tk.Label(self.pieCanvas, bg=self.secondary_background, fg=self.secondary_foreground, text=f"{int(100 * np.max(prediction)//1)}%", font=('Arial', 18, 'bold'))
        self.pred_label.grid(row=0, column=0, pady=(0,0), padx=(0, 0))

        desc_label=Image.open(self.description_image)
        desc_img=desc_label.resize((115, 33))
        self.desc_photo=ImageTk.PhotoImage(desc_img)

        self.desc_canvas = tk.Canvas(bg=self.secondary_background, width=115, height=33, border=0, highlightthickness=0)
        self.desc_canvas.grid(sticky=tk.NW, row=0, column=1, pady=(400,0), padx=(75,0))
        self.desc_canvas.create_image(0, 0, anchor=tk.NW, image=self.desc_photo)

        if len(predicted_class_label.split()) == 2:
            predicted_class_label = predicted_class_label.split()[0] + "\n" + predicted_class_label.split()[1]
        if len(predicted_class_label.split()) == 3:
            predicted_class_label = predicted_class_label.split()[0] + "\n" + predicted_class_label.split()[1] + " " + predicted_class_label.split()[2]
        if len(predicted_class_label) > 10:
            self.desc_label = tk.Label(bg=self.secondary_background, text=predicted_class_label.upper(), font=('Arial', 15, 'bold'), fg=self.secondary_foreground)
        else:
            self.desc_label = tk.Label(bg=self.secondary_background, text=predicted_class_label.upper(), font=('Arial', 15, 'bold'), fg=self.secondary_foreground)

        self.desc_label.grid(sticky=tk.NW ,row=0, column=1, pady=(450,0), padx=(75, 0))
        df = pd.read_excel(self.sport_description_file_url)
        text_row = df[df['id'] == predicted_class_index]
        text_desc = text_row['desc'].values[0]

        self.sport_desc_lbl = tk.Label(text=text_desc, bg=self.secondary_background, fg=self.secondary_foreground, font=('Arial', 12), wraplength=400, justify=tk.LEFT)
        self.sport_desc_lbl.grid(row=0, column=1, sticky=tk.N, pady=(520,0), padx=(25, 0))

    def change_cursor(self, event):
        self.button_canvas.config(cursor="hand2")

    def restore_cursor(self, event):
        self.button_canvas.config(cursor="")

    def load_model(self):
        script_directory = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_directory)
        self.model = tf.saved_model.load("./models/image_models/image_model/")

        if self.setting_image_loaded:
            self.classify_image(self.latest_image)

    def create_toggle_menu(self):
        menu_image=Image.open(self.toggle_menu_image)
        menu_img=menu_image.resize((50, 50))
        self.menu_photo=ImageTk.PhotoImage(menu_img)

        self.menu_btn_canvas = tk.Canvas(bg=self.primary_background, width=50, height=50, border=0, highlightthickness=0)
        self.menu_btn_canvas.grid(sticky=tk.NW, row=0, column=0, pady=(20,0), padx=(20,0))
        self.menu_btn_canvas.create_image(0, 0, anchor=tk.NW, image=self.menu_photo)
        self.menu_btn_canvas.bind("<Button-1>", self.toggle_menu)
        self.menu_btn_canvas.config(cursor="hand2")

    def toggle_menu(self, event):
        if self.language == 'cz':
            items = ['Obrázková klasifikace', 'Textová klasifikace', 'Nastavení', 'Nápověda', 'O autorovi', 'O aplikaci']
        else:
            items = ['Image classification', 'Text classification', 'Settings', 'Help', 'About the author', 'About the app']
        self.opened_menu = tk.Frame(self.root, bg=self.sidebar_background, width=250, height=800)
        self.opened_menu.grid(row=0, column=0, sticky=tk.W)

        self.opened_menu.grid_rowconfigure(0, weight=1)  # Keep the row at a fixed height

        close_menu_image = Image.open(self.close_menu_image)
        close_menu_img = close_menu_image.resize((50, 50))
        self.close_menu_photo = ImageTk.PhotoImage(close_menu_img)

        self.close_menu_btn_canvas = tk.Canvas(bg=self.sidebar_background, width=50, height=50, border=0, highlightthickness=0)
        self.close_menu_btn_canvas.grid(sticky=tk.NW, row=0, column=0, pady=(20, 0), padx=(20, 0))
        self.close_menu_btn_canvas.create_image(0, 0, anchor=tk.NW, image=self.close_menu_photo)
        self.close_menu_btn_canvas.bind("<Button-1>", self.close_menu)
        self.close_menu_btn_canvas.config(cursor="hand2")

        self.menu_item_labels = []

        for i, item_text in enumerate(items):
            menu_item = tk.Label(self.root, text=item_text, bg=self.sidebar_background, font=('Arial', 11, 'bold'), fg=self.sidebar_foreground)
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

        if ui_name == 'Obrázková klasifikace' or ui_name == 'Image classification':
            self.create_image_ui()
        elif ui_name == 'Textová klasifikace' or ui_name == 'Text classification':
            self.create_text_ui()
        elif ui_name == 'O aplikaci' or ui_name == 'About the app':
            self.create_about_ui()
        elif ui_name == 'O autorovi' or ui_name == 'About the author':
            self.create_me_ui()
        elif ui_name == 'Nápověda' or ui_name == 'Help':
            self.create_help_ui()
        elif ui_name == 'Nastavení' or ui_name == 'Settings':
            self.create_settings_ui()

    def on_focus(self, event):
        self.text_field.config(border=10)
        self.text_field.delete('1.0', 'end')

    def on_focus_out(self, event):
        self.text_field.config(border=1)
        
    def create_text_ui(self):
        self.get_settings()
        self.class_names = ["lukostřelba", "baseball", "basketbal", "kulečník", "bmx", "bowling", "box", "jízda na býku", "roztleskávání", "curling", "šerm", "krasobruslení", "americký fotbal", "závody formule 1", "golf", "skok do výšky", "hokej", "dostihy", "závody hydroplánů", "judo", "motocyklové závody", "pole dance", "rugby", "skoky na lyžích", "snowboarding", "rychlobruslení", "surfování", "plavání", "stolní tenis", "tenis", "dráhové kolo", "volejbal", "vzpírání"]
        
        self.grey_frame = tk.Frame(self.root, bg=self.primary_background, width=1050, height=800)
        self.grey_frame.grid(row=0, column=0)

        self.blue_frame = tk.Frame(self.root, bg=self.secondary_background, width=450, height=800)
        self.blue_frame.grid(row=0, column=1)

        script_directory = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_directory)

        self.create_toggle_menu()

        self.text_field = tk.Text(width=35, height=10, font=('Arial', 10), border=3, bg=self.secondary_background, fg=self.secondary_foreground)
        self.text_field.grid(sticky=tk.N, row=0, column=1, pady=(60,0))
        self.text_field.insert(tk.END, 'Describe the sport')
        self.text_field.bind("<FocusIn>", self.on_focus)
        self.text_field.bind("<FocusOut>", self.on_focus_out)
        
        imageLogo=Image.open(self.logo_image)
        imgLogo=imageLogo.resize((169, 33))
        self.photoLogo=ImageTk.PhotoImage(imgLogo)

        self.logo_canvas = tk.Canvas(bg=self.primary_background, width=169, height=33, border=0, highlightthickness=0)
        self.logo_canvas.grid(sticky=tk.SW, row=0, column=0, pady=(0,50), padx=(100,0))
        self.logo_canvas.create_image(0, 0, anchor=tk.NW, image=self.photoLogo)

        self.name_label = tk.Label(text="© 2023 Filip Cacák", bg=self.primary_background, fg="#d6d6d6", font=('Arial', 16, 'bold'))
        self.name_label.grid(sticky=tk.SW, row=0, column=0, pady=(0,50), padx=(290,0))
    
    def create_about_ui(self):
        if self.language == 'cz':
            self.class_names = ["lukostřelba", "baseball", "basketbal", "kulečník", "bmx", "bowling", "box", "jízda na býku", "roztleskávání", "curling", "šerm", "krasobruslení", "americký fotbal", "závody formule 1", "golf", "skok do výšky", "hokej", "dostihy", "závody hydroplánů", "judo", "motocyklové závody", "pole dance", "rugby", "skoky na lyžích", "snowboarding", "rychlobruslení", "surfování", "plavání", "stolní tenis", "tenis", "dráhové kolo", "volejbal", "vzpírání"]
            dic = [
                'O aplikaci',
                "SportSnapShot je aplikace vytvořená jako součást mého maturitního projektu, která využívá strojové učení pro klasifikaci sportů na základě obrázků a textových popisů. Aplikace vám umožní snadno a rychle zjistit, o jaký sport se jedná, a to jak na základě fotografií, tak i na základě textových popisů.",
                "Modely jsou aktuálně schopny rozpoznat 33 sportů, které naleznete v pravé části této obrazovky. Přesnější dokumentaci k modelům najdete ve zdrojovém kódu aplikace, nebo v mé maturitní práci"
            ]
        else:
            self.class_names = ["archery", "baseball", "basketball", "billiards", "bmx", "bowling", "boxing", "bull riding", "cheerleading", "curling", "fencing", "figure skating", "football", "formula 1", "golf", "high jump", "hockey", "horse racing", "hydroplane racing", "judo", "motorcycle racing", "pole dance", "rugby", "ski jumping", "snowboarding", "speed skating", "surfing", "swimming", "table tennis", "tennis", "track cycling", "volleyball", "weightlifting"]
            dic = [
                'About the app',
                "SportSnapShot is an app created as part of my senior project that uses machine learning to classify sports based on images and text descriptions. The app allows you to easily and quickly find out what sport it is based on both photos and text descriptions.",
                "The models are currently capable of recognising 33 sports, which can be found on the right-hand side of this screen. More precise documentation on the models can be found in the source code of the application or in my graduation thesis"
            ]
        
        self.grey_frame = tk.Frame(self.root, bg=self.primary_background, width=1050, height=800)
        self.grey_frame.grid(row=0, column=0)

        self.blue_frame = tk.Frame(self.root, bg=self.secondary_background, width=450, height=800)
        self.blue_frame.grid(row=0, column=1)

        script_directory = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_directory)

        self.create_toggle_menu()

        lbl = tk.Label(text=dic[0], font=('Arial', 20, 'bold'), bg=self.primary_background, fg="#53af32")
        lbl.grid(row=0, column=0, sticky=tk.N, pady=(20,0))

        text = tk.Label(fg=self.primary_foreground, text=dic[1], bg=self.primary_background, font=('Arial', 12), wraplength=700, justify=tk.LEFT)
        text.grid(row=0, column=0, sticky=tk.N, pady=(100,0))
        text1 = tk.Label(fg=self.primary_foreground, text=dic[2], bg=self.primary_background, font=('Arial', 12), wraplength=700, justify=tk.LEFT)
        text1.grid(row=0, column=0, sticky=tk.N, pady=(250,0))

        self.sport_item_labels = []

        for i, item in enumerate(self.class_names):
            sport_item = tk.Label(self.root, text=item, bg=self.secondary_background, fg=self.secondary_foreground, font=('Arial', 11, 'bold'))
            sport_item.grid(row=0, column=1, sticky=tk.NW, pady=(23*(i+1), 0), padx=(40, 0))

            self.sport_item_labels.append(sport_item)

    def create_me_ui(self):
        if self.language == 'cz':
            dic = ['O autorovi']
        else:
            dic = ['About the author']

        self.grey_frame = tk.Frame(self.root, bg=self.primary_background, width=1050, height=800)
        self.grey_frame.grid(row=0, column=0)

        self.blue_frame = tk.Frame(self.root, bg=self.secondary_background, width=450, height=800)
        self.blue_frame.grid(row=0, column=1)

        script_directory = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_directory)

        self.create_toggle_menu()

        lbl = tk.Label(text=dic[0], font=('Arial', 20, 'bold'), bg=self.primary_background, fg="#53af32")
        lbl.grid(row=0, column=0, sticky=tk.N, pady=(20,0))

        text = tk.Label(text="", bg=self.primary_background, font=('Arial', 12), wraplength=700, justify=tk.LEFT)
        text.grid(row=0, column=0, sticky=tk.N, pady=(100,0))
        text1 = tk.Label(text="", bg=self.primary_background, font=('Arial', 12), wraplength=700, justify=tk.LEFT)
        text1.grid(row=0, column=0, sticky=tk.N, pady=(250,0))

    def create_help_ui(self):
        if self.language == 'cz':
            dic = ['Nápověda']
        else:
            dic = ['Help']

        self.grey_frame = tk.Frame(self.root, bg=self.primary_background, width=1050, height=800)
        self.grey_frame.grid(row=0, column=0)

        self.blue_frame = tk.Frame(self.root, bg=self.secondary_background, width=450, height=800)
        self.blue_frame.grid(row=0, column=1)

        script_directory = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_directory)

        self.create_toggle_menu()

        lbl = tk.Label(text=dic[0], font=('Arial', 20, 'bold'), bg=self.primary_background, fg="#53af32")
        lbl.grid(row=0, column=0, sticky=tk.N, pady=(20,0))

        text = tk.Label(text="", bg=self.primary_background, font=('Arial', 12), wraplength=700, justify=tk.LEFT)
        text.grid(row=0, column=0, sticky=tk.N, pady=(100,0))
        text1 = tk.Label(text="", bg=self.primary_background, font=('Arial', 12), wraplength=700, justify=tk.LEFT)
        text1.grid(row=0, column=0, sticky=tk.N, pady=(250,0))
    
    def create_settings_ui(self):
        self.grey_frame = tk.Frame(self.root, bg=self.primary_background, width=1500, height=800)
        self.grey_frame.grid(row=0, column=0)
        if self.language == 'cz':
            dic = ['Nastavení', 'Tmavý režim', 'Jazyk']
        else:
            dic = ['Settings', 'Dark mode', 'Language']

        script_directory = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_directory)

        self.create_toggle_menu()

        #OBRAZOVKA
        lbl = tk.Label(text=dic[0], font=('Arial', 20, 'bold'), bg=self.primary_background, fg="#53af32")
        lbl.grid(row=0, column=0, sticky=tk.N, pady=(20,0))

        self.mode_switch_lbl = tk.Label(text=dic[1], bg=self.primary_background, font=('Arial', 12), fg=self.primary_foreground)
        self.mode_switch_lbl.grid(row=0, column=0, sticky=tk.N, pady=(100,0), padx=(0,70))
        
        if self.primary_background == 'white':
            self.mode_switch_btn = tk.Button(text='OFF', command=self.switch_mode, font=('Arial', 12), fg='#ff6666', bg=self.sidebar_background, border=0)
        else:
            self.mode_switch_btn = tk.Button(text='ON', command=self.switch_mode, font=('Arial', 12), fg='#53af32', bg=self.sidebar_background, border=0)

        self.mode_switch_btn.grid(row=0, column=0, sticky=tk.N, pady=(100,0), padx=(140,0))

        #JAZYK
        self.language_switch_lbl = tk.Label(text=dic[2], bg=self.primary_background, font=('Arial', 12), fg=self.primary_foreground)
        self.language_switch_lbl.grid(row=0, column=0, sticky=tk.N, pady=(160,0), padx=(0,70))
        
        if self.language == 'cz':
            self.language_switch_btn = tk.Button(text='CZ', command=self.switch_language, font=('Arial', 12), fg=self.primary_foreground, bg=self.sidebar_background, border=0)
        else:
            self.language_switch_btn = tk.Button(text='EN', command=self.switch_language, font=('Arial', 12), fg=self.primary_foreground, bg=self.sidebar_background, border=0)

        self.language_switch_btn.grid(row=0, column=0, sticky=tk.N, pady=(160,0), padx=(140,0))

    def switch_mode(self):
        script_directory = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_directory)
        parser = ConfigParser()
        parser.read('./config.ini')
        if parser.get('options', 'primary_background') == 'white':
            self.dark_mode(parser)
        else:
            self.light_mode(parser)

        self.get_settings()
        self.switch_ui('Nastavení')

    def switch_language(self):
        script_directory = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_directory)
        parser = ConfigParser()
        parser.read('./config.ini')
        if self.language == 'cz':
            self.en_language(parser)
        else:
            self.cz_language(parser)

        self.get_settings()
        self.switch_ui('Nastavení')

    def cz_language(self, parser):
        parser['options']['language'] = 'cz'
        with open('./config.ini', 'w') as configfile:
            parser.write(configfile)
        replacers = ['button_image', 'description_image', 'prediction_image', 'image_description_file']    
        for replacer in replacers:
            current = parser.get('options', replacer)
            new_value = current.replace('/en/', '/cz/')
            parser['options'][replacer] = new_value
            with open('./config.ini', 'w') as configfile:
                parser.write(configfile)
        
    def en_language(self, parser):
        parser['options']['language'] = 'en'
        with open('./config.ini', 'w') as configfile:
            parser.write(configfile)
        replacers = ['button_image', 'description_image', 'prediction_image', 'image_description_file']    
        for replacer in replacers:
            current = parser.get('options', replacer)
            new_value = current.replace('/cz/', '/en/')
            parser['options'][replacer] = new_value
            with open('./config.ini', 'w') as configfile:
                parser.write(configfile)

    def dark_mode(self, parser):
        parser['options']['primary_background'] = 'black'
        with open('./config.ini', 'w') as configfile:
            parser.write(configfile)
        parser['options']['primary_foreground'] = 'white'
        with open('./config.ini', 'w') as configfile:
            parser.write(configfile)
        parser['options']['secondary_background'] = '#1a1a1a'
        with open('./config.ini', 'w') as configfile:
            parser.write(configfile)
        parser['options']['secondary_foreground'] = 'white'
        with open('./config.ini', 'w') as configfile:
            parser.write(configfile)
        parser['options']['sidebar_background'] = '#1a1a1a'
        with open('./config.ini', 'w') as configfile:
            parser.write(configfile)
        parser['options']['sidebar_foreground'] = 'white'
        with open('./config.ini', 'w') as configfile:
            parser.write(configfile)
        parser['options']['button_image'] = './components/dark/'+ self.language +'/button.png'
        with open('./config.ini', 'w') as configfile:
            parser.write(configfile)
        parser['options']['toggle_menu_image'] = './components/dark/toggle_menu.png'
        with open('./config.ini', 'w') as configfile:
            parser.write(configfile)
        parser['options']['close_menu_image'] = './components/dark/close_menu.png'
        with open('./config.ini', 'w') as configfile:
            parser.write(configfile)

    def light_mode(self, parser):
        parser['options']['primary_background'] = 'white'
        with open('./config.ini', 'w') as configfile:
            parser.write(configfile)
        parser['options']['primary_foreground'] = 'black'
        with open('./config.ini', 'w') as configfile:
            parser.write(configfile)
        parser['options']['secondary_background'] = '#D9FFFF'
        with open('./config.ini', 'w') as configfile:
            parser.write(configfile)
        parser['options']['secondary_foreground'] = 'black'
        with open('./config.ini', 'w') as configfile:
            parser.write(configfile)
        parser['options']['sidebar_background'] = '#f5f5f5'
        with open('./config.ini', 'w') as configfile:
            parser.write(configfile)
        parser['options']['sidebar_foreground'] = 'black'
        with open('./config.ini', 'w') as configfile:
            parser.write(configfile)
        parser['options']['button_image'] = './components/light/'+ self.language +'/button.png'
        with open('./config.ini', 'w') as configfile:
            parser.write(configfile)
        parser['options']['toggle_menu_image'] = './components/light/toggle_menu.png'
        with open('./config.ini', 'w') as configfile:
            parser.write(configfile)
        parser['options']['close_menu_image'] = './components/light/close_menu.png'
        with open('./config.ini', 'w') as configfile:
            parser.write(configfile)

if __name__ == "__main__":
    ctypes.windll.shcore.SetProcessDpiAwareness(True)
    root = tk.Tk()
    app = ImageClassifierApp(root)
    app.load_model()
    root.mainloop()