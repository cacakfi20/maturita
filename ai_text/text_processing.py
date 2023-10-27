import unidecode
import re

stop_words = set("a, aby, aj, ale, anebo, ani, aniz, ano, asi, avska, az, ba, bez, bude, budem, budes, by, byl, byla, byli, bylo, byt, ci, clanek, clanku, clanky, co, com, coz, cz, dalsi, design, dnes, do, email, ho, i, jak, jake, jako, je, jeho, jej, jeji, jejich, jen, jeste, jenz, ji, jine, jiz, jsem, jses, jsi, jsme, jsou, jste, k, kam, kde, kdo, kdyz, ke, ktera, ktere, kteri, kterou, ktery, ku, ma, mate, me, mezi, mi, mit, mne, mnou, muj, muze, my, na, nad, nam, napiste, nas, nasi, ne, nebo, nebot, necht, nejsou, není, neni, net, nez, ni, nic, nove, novy, nybrz, o, od, ode, on, org, pak, po, pod, podle, pokud, pouze, prave, pred, pres, pri, pro, proc, proto, protoze, prvni, pta, re, s, se, si, sice, spol, strana, sve, svuj, svych, svym, svymi, ta, tak, take, takze, tamhle, tato, tedy, tema, te, ten, tedy, tento, teto, tim , timto, tipy, to, tohle, toho, tohoto, tom, tomto, tomuto, totiz, tu, tudiz, tuto, tvuj, ty, tyto, u, uz, v, vam, vas, vas, vase, ve, vedle, vice, vsak, vsechen, vy, vzdyt, z, za, zda, zde, ze, zpet, zpravy, mesto, dekuji, dobry, den".split(", "))

def preprocess_text(words):
    # conversion to lower case
    words = words.lower()
    # removal of markings
    words = unidecode.unidecode(words)
    words = re.sub(r'\d+', '', words)
    words = re.sub(r'\W+', ' ', words)
    # delete stop-slov
    words = remove_stopwords(words)
    words = remove_short_words(words)

    return words

def remove_stopwords(text):

    # Split the string into a list of words 
    words = text.split() 

    # Create a new list to hold the filtered words 
    filtered_words = [] 

    # Iterate over the list of words 
    for word in words: 
        # If the word is not in the stop word list, add it to the filtered list 
        if word not in stop_words: 
            filtered_words.append(word) 

    # Output: ['quick', 'brown', 'fox', 'jumps', 'lazy', 'dog.']
    return ' '.join(filtered_words)

def remove_short_words(text, min_length=3):

    # Split the string into a list of words 
    words = text.split() 

    # Create a new list to hold the filtered words 
    filtered_words = [] 

    # Iterate over the list of words 
    for word in words: 
        # If the word is not in the stop word list and its length is greater than or equal to the min_length, add it to the filtered list 
        if len(word) >= min_length: 
            filtered_words.append(word) 

    # Join the words back into a single string
    return ' '.join(filtered_words)
