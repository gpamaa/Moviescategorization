import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, validation_curve
from unidecode import unidecode

#Estrazione delle informazioni necessarie del file 'movies.csv'
def extract_movies():
    
    movies = pd.read_csv("dataset/movies.csv")
    # Caricamento del dataset
    movies_cleaned=movies.dropna()
    #eliminazione righe con valori nulli
    
    movies_cleaned['name'] = movies_cleaned['name'].str.replace(' ', '')
    movies_cleaned['director'] = movies_cleaned['director'].str.replace(' ', '')
    movies_cleaned['star'] = movies_cleaned['star'].str.replace(' ', '')
    movies_cleaned['writer'] = movies_cleaned['writer'].str.replace(' ', '')
    movies_cleaned['name'] = movies_cleaned['name'].str.replace("'", "")
    movies_cleaned['director'] = movies_cleaned['director'].str.replace("'", "")
    movies_cleaned['star'] = movies_cleaned['star'].str.replace("'", "")
    movies_cleaned['writer'] = movies_cleaned['writer'].str.replace("'", "")
    movies_cleaned['country']=movies_cleaned['country'].str.replace(' ', '')
    movies_cleaned['country'] = movies_cleaned['country'].str.replace("'", "")
    movies_cleaned['company']=movies_cleaned['company'].str.replace(' ', '')
    movies_cleaned['company']=movies_cleaned['company'].str.replace("'", "")
    movies_cleaned['company']=movies_cleaned['company'].str.replace('"', '')
    movies_cleaned['company']=movies_cleaned['company'].str.replace('.', '')
    movies_cleaned['name']=movies_cleaned['name'].str.replace('.', '')
    movies_cleaned['star'] = movies_cleaned['star'].str.replace('.', '')
    movies_cleaned['writer'] = movies_cleaned['writer'].str.replace('.', '')
    movies_cleaned['director'] = movies_cleaned['director'].str.replace(".", "")
    #eliminazione spazi e apici nei campi del dataset 
    
    movies_cleaned=movies_cleaned.drop("rating", axis=1)
    #eliminazione colonna raiting che serviva per distinguere i siti di raiting, che non è interessante per il progetto
    
    released = movies_cleaned['released'].str.split(' ', expand=True)
    movies_cleaned['month']=released[0]
    #assegno la prima parte della colonna released ad una nuova colonna che indica i mesi
   
    movies_cleaned['id'] = range(len(movies_cleaned))
     # Aggiungo una nuova colonna 'id' con valori incrementali
    
    movies_cleaned=movies_cleaned.drop("released",axis=1)
    #elimino la colonna released 

    movies_cleaned.replace('"','', inplace=True)
    movies_cleaned = movies_cleaned.applymap(lambda x: unidecode(x) if isinstance(x, str) else x)
    movies_cleaned.to_csv("dataset/working_dataset/movies.csv", index=False, mode='w')

    
def main():
    extract_movies()
    

main()