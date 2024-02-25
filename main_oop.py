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
from keras.preprocessing import image
from keras.models import load_model
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from ai_text.text_processing import preprocess_text
import pickle
import random
from sklearn.preprocessing import LabelEncoder
import sklearn

class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Image Classification - Sport edition')
        self.root.resizable(False, False)
        self.root.configure(bg='grey')
        self.create_image_ui()

    def get_settings(self):
        # Získání adresáře, ve kterém se nachází spouštěný skript
        script_directory = os.path.dirname(os.path.abspath(__file__))
        # Změna pracovního adresáře na adresář skriptu
        os.chdir(script_directory)
        # Inicializace objektu ConfigParser pro čtení konfiguračního souboru
        parser = ConfigParser()
        # Načtení konfiguračního souboru s názvem 'config.ini'
        parser.read('./config.ini')

        # Načtení různých nastavení z konfiguračního souboru pomocí metody get
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
        self.info_image = parser.get('options', 'info_image')

        self.language = parser.get('options', 'language')
        self.text_model_lang = parser.get('options', 'text_model_lang')
    
    def display_other_possibilities(self, event):
        # Vytvoření canvasu s určenou šířkou, výškou a nastavením okrajů a výplně
        self.other_canvas = tk.Canvas(width=500, height=200, border=0, highlightthickness=0)
        # Umístění canvasu na zvolenou pozici v hlavním okně
        self.other_canvas.grid(sticky=tk.NE, row=0, column=0, pady=(20,0), padx=(20,80))

        # Inicializace seznamu pro uchování popisků
        self.other_item_labels = []
        # Vytvoření hlavního nadpisu pro zobrazení nejbližších možností
        if self.language == 'cz':
            self.other_main = tk.Label(self.other_canvas, text="Nejbližší možnosti", font=('Arial', 14, 'bold'))
        else:
            self.other_main = tk.Label(self.other_canvas, text="Closest possibilities", font=('Arial', 14, 'bold'))
        # Umístění hlavního nadpisu na canvas
        self.other_main.grid(row=0, column=0, sticky=tk.N, pady=(0, 0))
        # Vytvoření popisků pro každou další možnost a jejich umístění na canvas
        for i, item_text in enumerate(self.other_possibilities):
            other_item = tk.Label(self.other_canvas, text=item_text[0]+': '+item_text[1]+'%', font=('Arial', 11, 'bold'))
            other_item.grid(row=0, column=0, sticky=tk.NW, pady=(30*(i+1), 0))
            self.other_item_labels.append(other_item)
    
    def destroy_other_possibilities(self, event):
        # Zrušení canvasu s možnostmi
        if hasattr(self, 'other_canvas'):
            self.other_canvas.destroy()
            for other_item in self.other_item_labels:
                other_item.destroy()
            self.other_item_labels = []

    def create_image_ui(self):
        # Načtení nastavení
        self.get_settings()
        
        # Seznam názvů sportů
        self.class_names = ["lukostřelba", "baseball", "basketbal", "kulečník", "bmx", "bowling", "box", "jízda na býku", "roztleskávání", "curling", "šerm", "krasobruslení", "americký fotbal", "závody formule 1", "golf", "skok do výšky", "hokej", "dostihy", "závody hydroplánů", "judo", "motocyklové závody", "pole dance", "rugby", "skoky na lyžích", "snowboarding", "rychlobruslení", "surfování", "plavání", "stolní tenis", "tenis", "dráhové kolo", "volejbal", "vzpírání"]
        
        # Vytvoření šedého rámečku
        self.grey_frame = tk.Frame(self.root, bg=self.primary_background, width=1050, height=800)
        self.grey_frame.grid(row=0, column=0)

        # Vytvoření modrého rámečku
        self.blue_frame = tk.Frame(self.root, bg=self.secondary_background, width=450, height=800)
        self.blue_frame.grid(row=0, column=1)

        # Získání adresáře, ve kterém se nachází spouštěný skript
        script_directory = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_directory)

        # Vytvoření rozbalovacího menu
        self.create_toggle_menu()
        
        # Pokus o zobrazení posledního obrázku
        try:
            # Otevření posledního obrázku
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

            # Zobrazení obrázku na plátně
            self.label = tk.Label(image=self.sport_photo)
            self.label.image = self.sport_photo
            self.label.grid(row=0, column=0)
            
            # Pokud je nastavený obrázek, provede se klasifikace
            if hasattr(self, 'setting_image_loaded'):
                self.classify_image(self.latest_image)
            
            # Zobrazení ikony informací na plátně
            info_image=Image.open(self.info_image)
            info_img=info_image.resize((50, 50))
            self.info_photo=ImageTk.PhotoImage(info_img)

            self.info_canvas = tk.Canvas(bg=self.primary_background, width=50, height=50, border=0, highlightthickness=0)
            self.info_canvas.grid(sticky=tk.NE, row=0, column=0, pady=(20,0), padx=(20,20))
            self.info_canvas.create_image(0, 0, anchor=tk.NW, image=self.info_photo)
            self.info_canvas.config(cursor="hand2")
            self.info_canvas.bind("<Enter>", self.display_other_possibilities)
            self.info_canvas.bind("<Leave>", self.destroy_other_possibilities)
            
            # Nastavení příznaku načtení obrázku
            self.setting_image_loaded = True
            # Klasifikace obrázku
            self.classify_image(self.latest_image)
        except:
            pass

        # Načtení tlačítka pro nahrání obrázku
        imagebtn=Image.open(self.button_image)
        imgbtn=imagebtn.resize((296, 183))
        self.photo=ImageTk.PhotoImage(imgbtn)

        self.button_canvas = tk.Canvas(bg=self.secondary_background, width=296, height=183, border=0, highlightthickness=0)
        self.button_canvas.grid(sticky=tk.N, row=0, column=1, pady=(60,0))
        self.button_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.button_canvas.bind("<Button-1>", self.upload_action)
        self.button_canvas.bind("<Enter>", self.change_cursor)
        self.button_canvas.bind("<Leave>", self.restore_cursor)

        # Načtení loga
        imageLogo=Image.open(self.logo_image)
        imgLogo=imageLogo.resize((169, 33))
        self.photoLogo=ImageTk.PhotoImage(imgLogo)

        self.logo_canvas = tk.Canvas(bg=self.primary_background, width=169, height=33, border=0, highlightthickness=0)
        self.logo_canvas.grid(sticky=tk.SW, row=0, column=0, pady=(0,50), padx=(100,0))
        self.logo_canvas.create_image(0, 0, anchor=tk.NW, image=self.photoLogo)

        # Vytvoření popisku s autorskými právy
        self.name_label = tk.Label(text="© 2023 Filip Cacák", bg=self.primary_background, fg="#d6d6d6", font=('Arial', 16, 'bold'))
        self.name_label.grid(sticky=tk.SW, row=0, column=0, pady=(0,50), padx=(290,0))

    def upload_action(self, event):
        # Zobrazení dialogu pro výběr souboru
        filename = filedialog.askopenfilename()
        # Otevření vybraného obrázku
        sport_img = Image.open(filename)
        max_width = 850
        max_height = 600
        width, height = sport_img.size
        width_scale = max_width / width
        height_scale = max_height / height
        scale_factor = min(width_scale, height_scale)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        # Změna velikosti obrázku
        sport_img = sport_img.resize((new_width, new_height))
        self.sport_photo = ImageTk.PhotoImage(sport_img)

        # Zničení předchozího zobrazení obrázku, pokud existuje
        if hasattr(self, 'label'):
            self.label.destroy()

        # Vytvoření nového zobrazení obrázku
        self.label = tk.Label(image=self.sport_photo)
        self.label.image = self.photo
        self.label.grid(row=0, column=0)
        # Klasifikace obrázku
        self.classify_image(filename)
        
        # Uložení cesty k poslednímu nahránému obrázku do konfiguračního souboru
        parser = ConfigParser()
        parser.read('./config.ini')
        parser['options']['latest_image'] = filename
        with open('./config.ini', 'w') as configfile:
            parser.write(configfile)
        
        # Aktualizace nastavení
        self.get_settings()

    def classify_image(self, img_url):
        # Načtení jazyka názvů
        if self.language == 'cz':
            self.class_names = ["lukostřelba", "baseball", "basketbal", "kulečník", "bmx", "bowling", "box", "jízda na býku", "roztleskávání", "curling", "šerm", "krasobruslení", "americký fotbal", "závody formule 1", "golf", "skok do výšky", "hokej", "dostihy", "závody hydroplánů", "judo", "motocyklové závody", "pole dance", "rugby", "skoky na lyžích", "snowboarding", "rychlobruslení", "surfování", "plavání", "stolní tenis", "tenis", "dráhové kolo", "volejbal", "vzpírání"]
        else:
            self.class_names = ["archery", "baseball", "basketball", "billiards", "bmx", "bowling", "boxing", "bull riding", "cheerleading", "curling", "fencing", "figure skating", "football", "formula 1", "golf", "high jump", "hockey", "horse racing", "hydroplane racing", "judo", "motorcycle racing", "pole dance", "rugby", "ski jumping", "snowboarding", "speed skating", "surfing", "swimming", "table tennis", "tennis", "track cycling", "volleyball", "weightlifting"]

        # Načtení obrázku
        img = image.load_img(img_url, target_size=(64, 64))
        X = image.img_to_array(img)
        X = np.expand_dims(X, axis=0)

        # Klasifikace obrázku
        prediction = self.model(X)
        predicted_class_index = tf.argmax(prediction, axis=1).numpy()[0]
        predicted_class_label = self.class_names[predicted_class_index]

        # Seřazení pravděpodobností
        top = tf.argsort(prediction, axis=-1, direction='DESCENDING')
        self.other_possibilities = []
        for i in top.numpy()[0][:3]:
            probability = prediction[0][i].numpy()
            self.other_possibilities.append([self.class_names[i], str(round(probability*100, 2))])
        
        # Zničení předchozího zobrazení obrázku, pokud existuje
        if hasattr(self, 'pred_label'):
            self.pred_label.destroy()   
        if hasattr(self, 'desc_label'):
            self.desc_label.destroy()
        if hasattr(self, 'sport_desc_lbl'):
            self.sport_desc_lbl.destroy()
        
        # Zobrazení obrázku s pravděpodobností
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

        # Zobrazení grafů a ostatních vizuálních efektů
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

        # Formátování textu
        if len(predicted_class_label.split()) == 2:
            predicted_class_label = predicted_class_label.split()[0] + "\n" + predicted_class_label.split()[1]
        if len(predicted_class_label.split()) == 3:
            predicted_class_label = predicted_class_label.split()[0] + "\n" + predicted_class_label.split()[1] + " " + predicted_class_label.split()[2]
        if len(predicted_class_label) > 10:
            self.desc_label = tk.Label(bg=self.secondary_background, text=predicted_class_label.upper(), font=('Arial', 12, 'bold'), fg=self.secondary_foreground)
        else:
            self.desc_label = tk.Label(bg=self.secondary_background, text=predicted_class_label.upper(), font=('Arial', 15, 'bold'), fg=self.secondary_foreground)

        self.desc_label.grid(sticky=tk.NW ,row=0, column=1, pady=(450,0), padx=(75, 0))
        df = pd.read_excel(self.sport_description_file_url)
        text_row = df[df['id'] == predicted_class_index]
        text_desc = text_row['desc'].values[0]

        self.sport_desc_lbl = tk.Label(text=text_desc, bg=self.secondary_background, fg=self.secondary_foreground, font=('Arial', 11), wraplength=400, justify=tk.LEFT)
        self.sport_desc_lbl.grid(row=0, column=1, sticky=tk.N, pady=(520,0), padx=(25, 0))

    def change_cursor(self, event):
        #nastaví kurzor na ruku
        self.button_canvas.config(cursor="hand2")

    def restore_cursor(self, event):
        # Resetuje kurzor
        self.button_canvas.config(cursor="")

    def model_load(self):
        # Načtení modelů strojového učení
        script_directory = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_directory)
        self.model = tf.saved_model.load("./models/image_models/image_model/")
        self.text_model_cz = load_model('./models/text_models/text_model_cz')
        #self.text_model_en = load_model('./models/text_models/text_model_en')

        if hasattr(self, 'setting_image_loaded'):
            self.classify_image(self.latest_image)

    def create_toggle_menu(self):
        #načtení obrázku pro tlačítko menu
        menu_image=Image.open(self.toggle_menu_image)
        menu_img=menu_image.resize((50, 50))
        self.menu_photo=ImageTk.PhotoImage(menu_img)

        #Vytvoření tlačítka pro otevření menu
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

        # Vytvoření menu
        self.opened_menu = tk.Frame(self.root, bg=self.sidebar_background, width=250, height=800)
        self.opened_menu.grid(row=0, column=0, sticky=tk.W)

        self.opened_menu.grid_rowconfigure(0, weight=1)

        close_menu_image = Image.open(self.close_menu_image)
        close_menu_img = close_menu_image.resize((50, 50))
        self.close_menu_photo = ImageTk.PhotoImage(close_menu_img)

        self.close_menu_btn_canvas = tk.Canvas(bg=self.sidebar_background, width=50, height=50, border=0, highlightthickness=0)
        self.close_menu_btn_canvas.grid(sticky=tk.NW, row=0, column=0, pady=(20, 0), padx=(20, 0))
        self.close_menu_btn_canvas.create_image(0, 0, anchor=tk.NW, image=self.close_menu_photo)
        self.close_menu_btn_canvas.bind("<Button-1>", self.close_menu)
        self.close_menu_btn_canvas.config(cursor="hand2")

        self.menu_item_labels = []
        # Vytvoření položek menu
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
        #zavře menu
        self.opened_menu.destroy()
        self.close_menu_btn_canvas.destroy()
        for menu_item in self.menu_item_labels:
            menu_item.destroy()
        self.menu_item_labels = []
    
    def switch_ui(self, ui_name):
        # Zničí ostatní atributy v aplikaci 
        for child in root.winfo_children():
            child.destroy()

        # Vytvoří nové uživatelské rozhraní podle vybrání v menu
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
        # Nastaví větší okraj textového pole při kliknutí
        self.text_field.config(border=10)
        self.text_field.delete('1.0', 'end')

    def on_focus_out(self, event):
        # Nastaví menší okraj textového pole při kliknutí mimo něj
        self.text_field.config(border=1)
    
    def switch_text_model(self):
        # Přepne jazyk textového modelu
        parser = ConfigParser()
        parser.read('./config.ini')
        if self.text_model_lang == 'cz':
            parser['options']['text_model_lang'] = 'en'
        else:
            parser['options']['text_model_lang'] = 'cz'
        with open('./config.ini', 'w') as configfile:
            parser.write(configfile)
        self.get_settings()
        if self.text_model_lang == 'cz':
            print('přepnuto na český')
        else:
            print('přepnuto na anglický režim')
        self.create_text_ui()
    
    def create_text_ui(self):
        # Načtení nastavení
        self.get_settings()
        # Seznam názvů sportů
        self.class_names = ["lukostřelba", "baseball", "basketbal", "kulečník", "bmx", "bowling", "box", "jízda na býku", "roztleskávání", "curling", "šerm", "krasobruslení", "americký fotbal", "závody formule 1", "golf", "skok do výšky", "hokej", "dostihy", "závody hydroplánů", "judo", "motocyklové závody", "pole dance", "rugby", "skoky na lyžích", "snowboarding", "rychlobruslení", "surfování", "plavání", "stolní tenis", "tenis", "dráhové kolo", "volejbal", "vzpírání"]
        self.en_class_names = ["archery", "baseball", "basketball", "billiards", "bmx", "bowling", "boxing", "bull riding", "cheerleading", "curling", "fencing", "figure skating", "football", "formula 1 racing", "golf", "high jump", "hockey", "horse racing", "hydroplane racing", "judo", "motorcycle racing", "pole dance", "rugby", "ski jumping", "snow boarding", "speed skating", "surfing", "swimming", "table tennis", "tennis", "track cycling", "volleyball", "weightlifting"]
    
        if self.language == 'cz':
            disc = ['Popiš sport', 'Odeslat', 'Nastav jazyk textového modelu']
        else:
            disc = ['Describe the sport', 'Submit', 'Set the language of the text model']
        # Vytvoření hlavního rámečku
        self.grey_frame = tk.Frame(self.root, bg=self.primary_background, width=1050, height=800)
        self.grey_frame.grid(row=0, column=0)

        # Vytvoření vedlejšího rámečku
        self.blue_frame = tk.Frame(self.root, bg=self.secondary_background, width=450, height=800)
        self.blue_frame.grid(row=0, column=1)

        script_directory = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_directory)

        # Zavře menu
        self.create_toggle_menu()

        '''self.language_switch_lbl = tk.Label(text=disc[2], bg=self.primary_background, font=('Arial', 12), fg=self.primary_foreground)
        self.language_switch_lbl.grid(row=0, column=1, sticky=tk.N, pady=(20,0), padx=(0,0))
        
        # Tlačítka na změnu jazyka modelu
        if self.text_model_lang == 'cz':
            self.language_switch_btn = tk.Button(text='CZ', command=self.switch_text_model, font=('Arial', 12), fg=self.primary_foreground, bg=self.primary_background, border=0)
        else:
            self.language_switch_btn = tk.Button(text='EN', command=self.switch_text_model, font=('Arial', 12), fg=self.primary_foreground, bg=self.primary_background, border=0)

        self.language_switch_btn.grid(row=0, column=1, sticky=tk.N, pady=(55,0), padx=(0,0))
        self.language_switch_btn.config(cursor="hand2")
        '''
        # Vytvoření textového pole
        self.text_field = tk.Text(width=35, height=10, font=('Arial', 10), border=3, bg=self.secondary_background, fg=self.secondary_foreground, insertbackground=self.secondary_foreground)
        self.text_field.grid(sticky=tk.N, row=0, column=1, pady=(90,0))
        self.text_field.insert(tk.END, disc[0])
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
    
        self.submit_btn = tk.Button(text=disc[1], bg=self.secondary_background, fg=self.secondary_foreground, font=('Arial', 12, 'bold'), border=0, highlightthickness=0, command=self.classify_text)
        self.submit_btn.grid(sticky=tk.N, row=0, column=1, pady=(230,0))
        self.submit_btn.config(cursor="hand2")
    
    def classify_text(self):
        # Vezme text z textového pole
        self.input = self.text_field.get("1.0", 'end-1c')
        print(self.input)
        # Načte encodery otřebné pro klasifikaci
        with open('./ai_text/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        with open('./ai_text/encoder.pickle', 'rb') as handle:
            encoder = pickle.load(handle)
            string = preprocess_text(self.input)
        # Převede text na sekvenci
        sequences = tokenizer.texts_to_sequences([string])
        # Připravuje text na základě jazyka
        if self.text_model_lang == 'en':
            data = pad_sequences(sequences, maxlen=38)
        else:
            data = pad_sequences(sequences, maxlen=28)

        # Klasifikuje text na záklaě jazyka
        if self.text_model_lang == 'cz':
            predictions = self.text_model_cz.predict(data)
        else:
            predictions = self.text_model_en.predict(data)
        
        # Převede predikce na názvy sportů
        predicted_labels = predictions.argmax(axis=-1)
        predicted_labels = encoder.inverse_transform(predicted_labels)
        predicted_confidence = predictions.max(axis=-1)
        print(self.class_names[predicted_labels[0]], predicted_confidence[0])
        self.get_random_picture(self.en_class_names[predicted_labels[0]])
        if hasattr(self, 'pred_label'):
            self.pred_label.destroy()   
        if hasattr(self, 'desc_label'):
            self.desc_label.destroy()
        if hasattr(self, 'sport_desc_lbl'):
            self.sport_desc_lbl.destroy()
        
        script_directory = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_directory)

        # Vytvoření vizuálních efektů a ostatních věcí
        self.pieCanvas = tk.Canvas(bg=self.secondary_background, width=296, height=296, border=0, highlightthickness=0)
        self.pieCanvas.grid(sticky=tk.NE, row=0, column=1, pady=(360,0), padx=(0, 27))

        fig = matplotlib.figure.Figure(figsize=(1.5,1.5))
        fig.patch.set_facecolor(self.secondary_background)
        ax = fig.add_subplot(111)
        if 100 * np.max(predicted_confidence)//1 <= 30:
            ax.pie([100 * np.max(predicted_confidence)//1,100-100 * np.max(predicted_confidence)//1], colors=["#ff6666",self.secondary_background], startangle=90)   
        if 100 * np.max(predicted_confidence)//1 > 30:
            ax.pie([100 * np.max(predicted_confidence)//1,100-100 * np.max(predicted_confidence)//1], colors=["#e68541",self.secondary_background], startangle=90)    
        if 100 * np.max(predicted_confidence)//1 >= 70:
            ax.pie([100 * np.max(predicted_confidence)//1,100-100 * np.max(predicted_confidence)//1], colors=["#53af32",self.secondary_background], startangle=90) 
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

        self.pred_label = tk.Label(self.pieCanvas, bg=self.secondary_background, fg=self.secondary_foreground, text=f"{int(100 * np.max(predicted_confidence)//1)}%", font=('Arial', 18, 'bold'))
        self.pred_label.grid(row=0, column=0, pady=(0,0), padx=(0, 0))

        desc_label=Image.open(self.description_image)
        desc_img=desc_label.resize((115, 33))
        self.desc_photo=ImageTk.PhotoImage(desc_img)

        self.desc_canvas = tk.Canvas(bg=self.secondary_background, width=115, height=33, border=0, highlightthickness=0)
        self.desc_canvas.grid(sticky=tk.NW, row=0, column=1, pady=(400,0), padx=(75,0))
        self.desc_canvas.create_image(0, 0, anchor=tk.NW, image=self.desc_photo)

        predicted_class_label = self.class_names[predicted_labels[0]]

        if len(predicted_class_label.split()) == 2:
            predicted_class_label = predicted_class_label.split()[0] + "\n" + predicted_class_label.split()[1]
        if len(predicted_class_label.split()) == 3:
            predicted_class_label = predicted_class_label.split()[0] + "\n" + predicted_class_label.split()[1] + " " + predicted_class_label.split()[2]
        if len(predicted_class_label) > 10:
            self.desc_label = tk.Label(bg=self.secondary_background, text=predicted_class_label.upper(), font=('Arial', 12, 'bold'), fg=self.secondary_foreground)
        else:
            self.desc_label = tk.Label(bg=self.secondary_background, text=predicted_class_label.upper(), font=('Arial', 15, 'bold'), fg=self.secondary_foreground)

        self.desc_label.grid(sticky=tk.NW ,row=0, column=1, pady=(450,0), padx=(75, 0))
        df = pd.read_excel(self.sport_description_file_url)
        text_row = df[df['id'] == predicted_labels[0]]
        text_desc = text_row['desc'].values[0]

        self.sport_desc_lbl = tk.Label(text=text_desc, bg=self.secondary_background, fg=self.secondary_foreground, font=('Arial', 11), wraplength=400, justify=tk.LEFT)
        self.sport_desc_lbl.grid(row=0, column=1, sticky=tk.N, pady=(520,0), padx=(25, 0))

    def get_random_picture(self, sport):
        # Načte náhodný obrázek sportu
        randomint = random.randint(1, 5) 
        sport_img = Image.open('./data/test/'+sport+'/'+str(randomint)+'.jpg')
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
        
        # Zobrazí obrázek
        self.label = tk.Label(image=self.sport_photo)
        self.label.image = self.sport_photo
        self.label.grid(row=0, column=0)

    def create_about_ui(self):
        # Vytvoří slovníky
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
        
        # Zobrazení rámečků a textů
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

        # Vytvoření seznamu sportů v aplikaci
        for i, item in enumerate(self.class_names):
            sport_item = tk.Label(self.root, text=item, bg=self.secondary_background, fg=self.secondary_foreground, font=('Arial', 11, 'bold'))
            sport_item.grid(row=0, column=1, sticky=tk.NW, pady=(23*(i+1), 0), padx=(40, 0))

            self.sport_item_labels.append(sport_item)

    def create_me_ui(self):
        # Vytvoření slovníku
        if self.language == 'cz':
            dic = ['O autorovi']
        else:
            dic = ['About the author']

        self.grey_frame = tk.Frame(self.root, bg=self.primary_background, width=1500, height=800)
        self.grey_frame.grid(row=0, column=0)

        script_directory = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_directory)

        self.create_toggle_menu()

        # Zobrazení textu
        lbl = tk.Label(text=dic[0], font=('Arial', 20, 'bold'), bg=self.primary_background, fg="#53af32")
        lbl.grid(row=0, column=0, sticky=tk.N, pady=(20,0))

        if self.language == 'cz':
            text = tk.Label(text="Jmenuji se Filip Cacák a jsem studentem 4. ročníku střední průmyslové školy na Proseku (2023/24), zároveň jsem tvůrce tohoto maturitního projektu. Mimo školu pracuji na dohodu ve firmě Operátor ICT, kde se zabývám především vývojem a úpravou frontendu a backendu webovým aplikací pro Prahu a středočeský kraj. Mimo jiné také rád sportuji a jsem už od mala velkým fanouškem fotbalu, jelikož jsem ho celý život hrál. O víkendech rád trávím volný čas s přáteli a rodinou.", bg=self.primary_background, fg=self.primary_foreground, font=('Arial', 12), wraplength=700, justify=tk.LEFT)
        else:
            text = tk.Label(text="My name is Filip Cacák and I am a student of the 4th year of the Secondary Industrial School in Prosek (2023/24), I am also the creator of this graduation project. Outside of school, I work on contract in the company Operátor ICT, where I mainly deal with the development and modification of frontend and backend web applications for Prague and Central Bohemia. Among other things, I also like sports and have been a big fan of football since I was a kid, as I have played it all my life. On weekends I like to spend my free time with friends and family.", bg=self.primary_background, fg=self.primary_foreground, font=('Arial', 12), wraplength=700, justify=tk.LEFT)            
        text.grid(row=0, column=0, sticky=tk.N, pady=(100,0))
        text1 = tk.Label(text="", bg=self.primary_background, font=('Arial', 12), wraplength=700, justify=tk.LEFT)
        text1.grid(row=0, column=0, sticky=tk.N, pady=(250,0))

        youngphoto_img = Image.open('./components/malycacak.jpg')
        youngphoto_img = youngphoto_img.resize((400,320))
        self.youngphoto_photo = ImageTk.PhotoImage(youngphoto_img)
        # Zobrazí obrázek
        self.label1 = tk.Label(image=self.youngphoto_photo)
        self.label1.image = self.youngphoto_photo
        self.label1.grid(row=0, column=0, sticky=tk.SW, pady=(0, 50), padx=(50,0))

        olderphoto_img = Image.open('./components/strednicacak.jpg')
        olderphoto_img = olderphoto_img.resize((480,320))
        self.olderphoto_photo = ImageTk.PhotoImage(olderphoto_img)
        # Zobrazí obrázek
        self.label2 = tk.Label(image=self.olderphoto_photo)
        self.label2.image = self.olderphoto_photo
        self.label2.grid(row=0, column=0, sticky=tk.S, pady=(0, 50), padx=(0, 80))

        bigphoto_img = Image.open('./components/velkycacak.jpg')
        bigphoto_img = bigphoto_img.resize((480,320))
        self.bigphoto_photo = ImageTk.PhotoImage(bigphoto_img)
        # Zobrazí obrázek
        self.label2 = tk.Label(image=self.bigphoto_photo)
        self.label2.image = self.bigphoto_photo
        self.label2.grid(row=0, column=0, sticky=tk.SE, pady=(0, 50), padx=(0,50))


    def create_help_ui(self):
        # Vytvoření slovníku
        if self.language == 'cz':
            dic = ['Nápověda', 'Obrázkový model', 'Textový model']
        else:
            dic = ['Help', 'Image model', 'Text model']

        self.grey_frame = tk.Frame(self.root, bg=self.primary_background, width=1500, height=800)
        self.grey_frame.grid(row=0, column=0)

        script_directory = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_directory)

        self.create_toggle_menu()

        # Zobrazení textů
        lbl = tk.Label(text=dic[0], font=('Arial', 20, 'bold'), bg=self.primary_background, fg="#53af32")
        lbl.grid(row=0, column=0, sticky=tk.N, pady=(20,0))
        lbl_image_model = tk.Label(text=dic[1], bg=self.primary_background, fg=self.primary_foreground, font=('Arial', 14), wraplength=700, justify=tk.LEFT)
        lbl_image_model.grid(row=0, column=0, sticky=tk.NW, pady=(100,0), padx=(150,0))

        if self.language == 'cz':
            text = tk.Label(text="1) Nahraje fotografii sportu tlačítkem vpravo nahoře\n2) Model fotografii vyhodnotí a zobrazí klasifikaci s bližším popisem sportu", bg=self.primary_background, fg=self.primary_foreground, font=('Arial', 12), wraplength=700, justify=tk.LEFT)    
        else:
            text = tk.Label(text="1) Upload a photo of the sport using the button on the top right\n2) The model evaluates the photo and displays the classification with closer sport description", bg=self.primary_background, fg=self.primary_foreground, font=('Arial', 12), wraplength=700, justify=tk.LEFT)

        text.grid(row=0, column=0, sticky=tk.NW, pady=(140,0), padx=(150,0))

        lbl_text_model = tk.Label(text=dic[2], bg=self.primary_background, fg=self.primary_foreground, font=('Arial', 14), wraplength=700, justify=tk.LEFT)
        lbl_text_model.grid(row=0, column=0, sticky=tk.NW, pady=(240,0), padx=(150,0))

        if self.language == 'cz':
            text1 = tk.Label(text="1) Napiš text do textového pole pravo nahoře a klikani na tlačítko\n2) Model text vyhodnotí a zobrazí klasifikaci s bližším popisem sportu a jeho obrázkem", bg=self.primary_background, fg=self.primary_foreground, font=('Arial', 12), wraplength=700, justify=tk.LEFT)    
        else:
            text1 = tk.Label(text="1) Type the text in the text box on the top right and click on the button\n2) The model will evaluate the text and display the classification with a more detailed description of the sport and its picture", bg=self.primary_background, fg=self.primary_foreground, font=('Arial', 12), wraplength=700, justify=tk.LEFT)

        text1.grid(row=0, column=0, sticky=tk.NW, pady=(280,0), padx=(150,0))
    
    def create_settings_ui(self):
        self.grey_frame = tk.Frame(self.root, bg=self.primary_background, width=1500, height=800)
        self.grey_frame.grid(row=0, column=0)
        # Vytvoření slovníku
        if self.language == 'cz':
            dic = ['Nastavení', 'Tmavý režim', 'Jazyk']
        else:
            dic = ['Settings', 'Dark mode', 'Language']

        script_directory = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_directory)

        self.create_toggle_menu()

        # Nastavení tlačítka pro změnu režimu (tmavý/světlý)
        lbl = tk.Label(text=dic[0], font=('Arial', 20, 'bold'), bg=self.primary_background, fg="#53af32")
        lbl.grid(row=0, column=0, sticky=tk.N, pady=(20,0))

        self.mode_switch_lbl = tk.Label(text=dic[1], bg=self.primary_background, font=('Arial', 12), fg=self.primary_foreground)
        self.mode_switch_lbl.grid(row=0, column=0, sticky=tk.N, pady=(100,0), padx=(0,70))
        
        if self.primary_background == 'white':
            self.mode_switch_btn = tk.Button(text='OFF', command=self.switch_mode, font=('Arial', 12), fg='#ff6666', bg=self.sidebar_background, border=0)
        else:
            self.mode_switch_btn = tk.Button(text='ON', command=self.switch_mode, font=('Arial', 12), fg='#53af32', bg=self.sidebar_background, border=0)

        self.mode_switch_btn.grid(row=0, column=0, sticky=tk.N, pady=(100,0), padx=(140,0))

        # Nastavení tlačítka pro změnu jazyka
        self.language_switch_lbl = tk.Label(text=dic[2], bg=self.primary_background, font=('Arial', 12), fg=self.primary_foreground)
        self.language_switch_lbl.grid(row=0, column=0, sticky=tk.N, pady=(160,0), padx=(0,70))
        
        if self.language == 'cz':
            self.language_switch_btn = tk.Button(text='CZ', command=self.switch_language, font=('Arial', 12), fg=self.primary_foreground, bg=self.sidebar_background, border=0)
        else:
            self.language_switch_btn = tk.Button(text='EN', command=self.switch_language, font=('Arial', 12), fg=self.primary_foreground, bg=self.sidebar_background, border=0)

        self.language_switch_btn.grid(row=0, column=0, sticky=tk.N, pady=(160,0), padx=(140,0))

    def switch_mode(self):
        # Přepne varevný režim aplikace
        script_directory = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_directory)
        parser = ConfigParser()
        parser.read('./config.ini')
        # Uloží režim do nastavení
        if parser.get('options', 'primary_background') == 'white':
            self.dark_mode(parser)
        else:
            self.light_mode(parser)

        self.get_settings()
        self.switch_ui('Nastavení')

    def switch_language(self):
        # Přepne jazyk aplikace
        script_directory = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_directory)
        parser = ConfigParser()
        parser.read('./config.ini')
        # Uloží jazyk do nastavení
        if self.language == 'cz':
            self.en_language(parser)
        else:
            self.cz_language(parser)

        self.get_settings()
        self.switch_ui('Nastavení')

    def cz_language(self, parser):
        # Nastavení českého jazyka
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
        # Nastavení anglického jazyka
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
        # Nastavení tmavého režimu
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
        parser['options']['info_image'] = './components/dark/more_info.png'
        with open('./config.ini', 'w') as configfile:
            parser.write(configfile)

    def light_mode(self, parser):
        # Nastavení světlého režimu
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
        parser['options']['info_image'] = './components/light/more_info.png'
        with open('./config.ini', 'w') as configfile:
            parser.write(configfile)

if __name__ == "__main__":
    ctypes.windll.shcore.SetProcessDpiAwareness(True)
    root = tk.Tk()
    app = ImageClassifierApp(root)
    app.model_load()
    root.mainloop()
