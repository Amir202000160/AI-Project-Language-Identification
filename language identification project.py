import pandas as pd
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import speech_recognition as sr 
import tkinter as tk
from PIL import Image, ImageTk
import csv

print("Welcome To The Language Identifier")

print("PLease Select your an option: ")

print("Enter  1  if you Choose to identify a text Query")
print("Enter  2  if you Choose to input identify a voice Query")

opt = int(input("Enter Your number : "))
if (opt!=1 and opt!=2):

    print("Program exit due to Wrong intput")
    exit(0)
else:
    stopwords = nltk.corpus.stopwords.words('english')
    ps = nltk.PorterStemmer()

    def clean_text(text):

        toke=list(text.split(' '))
        
        t_lator=str.maketrans('','',string.punctuation)
        text=text.translate(t_lator)
        remove_digits = str.maketrans('', '', string.digits)
        text = text.translate(remove_digits)
        
        for i in '“”—':
            text = text.replace(i, ' ')

        return text

    def clean_data(df):
        df.dropna(how='any')

    vectorizer = TfidfVectorizer()
    def train_data(df):

        X = vectorizer.fit_transform(df['text'])
        true_k = 4
        model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
        model.fit(X)
        order_centroids = model.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names_out()
        for i in range(true_k):
            print("Cluster %d:" % i),
            for ind in order_centroids[i, :10]:
                print(' %s' %terms[ind]),
                
        return model

    print("_______________________________________________________")
    print("All languages in The Program ")
    print("_______________________________________________________")
    df = pd.read_csv("D:\VSCODE\App\Langident\dataset.csv")
    df.dropna(how='any') 
    df.columns=['text','language']
    df = df.sort_values(['language'])
    i = 0
    while i < 4000:
         print(df.iat[i, 1])
         i += 1000
    print("_______________________________________________________")
    print("Learning from the dataset..... ")
    print("_______________________________________________________")
    clean_data(df)
    train=train_data(df)
    if(opt == 2 ):
        print("_______________________________________________________")
        print("voice input ")
        print("_______________________________________________________")

        r = sr.Recognizer()
        with sr.Microphone() as source:
                print("speak")
                audio= r.record(source, duration=5)
        try:
            print("Recognising.....")
            text= r.recognize_google(audio)
            print('{}'.format(text))
            clean_text(text)
        except:
            print("Try Again!")
        clean_text(text)    

    if(opt == 1 ):
        print("_______________________________________________________")
        print("Text Input ")
        print("_______________________________________________________")

        text = input("Please Enter The Text :\n")
        clean_text(text)
    def predicti (model):
        try:
            Y = vectorizer.transform([text])
            prediction = model.predict(Y)
            print (prediction)

            z= prediction[0]*1000
            return df.iat[z, 1]
        except:
            print("Try Again!")

    print("_______________________________________________________")
    print("Model Prediction")
    print("_______________________________________________________")
    try:
        output = predicti(train)
        print("The Predicted Language is : " + output)
    except:
        print("Try Again!")

    root = tk.Tk()
    root.title("Automatic Language Identification using K-means Clustering")
    root.geometry("600x600")

    if(output == 'Arabic'):
        photo = Image.open("D:\VSCODE\App\Langident\Images\Arabic.jpg")
        resized_image = photo.resize((600,600), Image.ANTIALIAS)
        converted_image = ImageTk.PhotoImage(resized_image)
    elif(output == 'English'):
        photo = Image.open("D:\VSCODE\App\Langident\Images\English.jpg")
        resized_image = photo.resize((600,600), Image.ANTIALIAS)
        converted_image = ImageTk.PhotoImage(resized_image)
    elif(output == 'French'):
        photo = Image.open("D:\VSCODE\App\Langident\Images\French.jpg")
        resized_image = photo.resize((600,600), Image.ANTIALIAS)
        converted_image = ImageTk.PhotoImage(resized_image)
    elif(output == 'Dutch'):
        photo = Image.open("D:\VSCODE\App\Langident\Images\Dutch.jpg")
        resized_image = photo.resize((600,600), Image.ANTIALIAS)
        converted_image = ImageTk.PhotoImage(resized_image)
    elif(output == 'Chinese'):
        photo = Image.open("D:\VSCODE\App\Langident\Images\Chinese.jpg")
        resized_image = photo.resize((600,600), Image.ANTIALIAS)
        converted_image = ImageTk.PhotoImage(resized_image)    
    else:
        label = tk.Label(root, text="Sorry try again", width=600, height=600)

try:
    label = tk.Label(root, image = converted_image, width = 600 , height = 600)
    label.pack()
    root.eval('tk::PlaceWindow . center')
    root.attributes('-topmost',True)
    root.mainloop()
except:
    print("try again!")
   
   
