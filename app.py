from deep_translator import GoogleTranslator
from flask import Flask, render_template, request, jsonify
import base64
import speech_recognition as sr
from gtts import gTTS
from easyocr import Reader
from tr_func import *
from PIL import Image
import io
#from aud_func import *
import pyaudio
import easyocr

app = Flask(__name__)
UPLOAD_FOLDER='static/'
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER
# Home route
@app.route('/')
def home():
    return render_template('index.html')



@app.route('/translate1', methods=['POST'])
def translate1():
    text = request.form['text1']
    #language = request.form['language']
    print(text)
    result=translateText(text)
    print(result)
    # Return the translated result
    #translated_text = "Translated text goes here"  # Replace with your translation code
    return render_template('index.html',t1=text, r1=result)


@app.route('/translate', methods=['POST'])
def translate():
    text = request.form['text']
    source_language = request.form['language_from']
    target_language = request.form['language_to']
    print(text)
    print(target_language)

    translated_text = GoogleTranslator(source=source_language, target=target_language).translate(text)
    response = {
        'translated_text': translated_text
    }
    

    return render_template('index.html',t=text, r=translated_text)

@app.route('/translate_aud', methods=['POST'])
def translate_aud():
    source_language = request.form['language_aud_from']
    target_language = request.form['language_aud']
    print(source_language)
    # Enregistrer l'audio à partir du microphone
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Enregistrement audio...")
        audio = recognizer.listen(source,timeout=5)

    # Convertir l'audio en texte
    if(source_language=='fr'):
        text = recognizer.recognize_google(audio, language="fr-FR")
    
    if(source_language=='en'):
        text = recognizer.recognize_google(audio, language="en-US")
    if(source_language=='ar'):
        text = recognizer.recognize_google(audio, language="ar-SA")
    print(text)
    
    # Effectuer la traduction
    # Traduire le texte
    translated_text = GoogleTranslator(source=source_language, target=target_language).translate(text)
    response = {
        'translated_text': translated_text
    }

    return render_template('index.html', t=text, r2=translated_text)

@app.route('/translate_img', methods=['POST'])
def translate_img():
    # Récupérer l'image depuis la requête
    image_data = request.files['image']
    
    image_path = 'static/image.jpg'  # Specify the temporary image file path
    image_data.save(image_path)

    # Charger l'image et extraire le texte avec easyocr
    #image = Image.open(image_stream)
    reader = easyocr.Reader(['en'])  # Indiquez les langues nécessaires ici
    result = reader.readtext(image_path)
    
    # Concaténer le texte extrait en une seule chaîne
    text = ' '.join([item[1] for item in result])
    
    # Récupérer la langue cible depuis le formulaire
    source_language = request.form['language_img_from']
    target_language = request.form['language_img']
    
    # Traduire le texte
    translated_text = GoogleTranslator(source=source_language, target=target_language).translate(text)
    response = {
        'translated_text': translated_text
    }
    
    return render_template('index.html', t3=text, r3=translated_text)


    #return render_template('index.html', t3=transcript, r3=translated_text)

if __name__ == '__main__':
    app.run(debug=True)
