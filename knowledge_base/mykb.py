import time
import pandas as pd
from pyswip import Prolog
import numpy as np
#creazione delle clausole
def create_kb() -> Prolog:
    prolog = Prolog()

    prolog.consult("knowledge_base/movies.pl")
    kb=prolog

    #calcolo del guadagno
    prolog.assertz("get_movie_earn_perc(Id,Earn) :- movie(Id,_,_,_,_,_,_,_,_,_,Budget,Gross,_,_,_),Earn is (((Gross-Budget)/Budget)*100) ") #restituisce il valore in percentuale del guadagno
    prolog.assertz("get_movie_earn(Id,Earn) :- movie(Id,_,_,_,_,_,_,_,_,_,Budget,Gross,_,_,_),Earn is Gross-Budget ")#restituisce il valore del guadagno    
    #calcolo categoria
    prolog.assertz("get_movie_category(Id, Category) :- get_movie_earn_perc(Id, Earn), (Earn > 50, Category = 'masterpiece'; Earn >= 25, Earn =< 50, Category = 'success'; Earn < 25, Earn >= 0, Category = 'flop'; Earn < 0, Category = 'big_flop')")#restituisce la categoria del film in base al guadagno
    prolog.assertz("apply_get_movie_category(_, [], [])")#caso base lista vuota
    prolog.assertz("apply_get_movie_category(get_movie_category,[Element|Rest], [Result|RestResult]):- call(get_movie_category,Element,Result), apply_get_movie_category(get_movie_category,Rest,RestResult)") #funzione ricorsiva per ottenere per ogni film della lista la categoria
    prolog.assertz("get_movies_category(MoviesListS):- findall(Id,movie(Id,_,_,_,_,_,_,_,_,_,_,_,_,_,_),Movies),apply_get_movie_category(get_movie_category,Movies,MoviesList),convert_list_to_string(MoviesList,MoviesListS)")#funzione per calcolare per tutti i film la categoria
    
    #calcolo guadagni film dal nome del regista
    prolog.assertz("apply_get_movie_earn(_, [], [])")  #funzione ricorsiva Caso base: la lista è vuota
    prolog.assertz("apply_get_movie_earn(get_movie_earn,[Element|Rest], [Result|RestResult]):- call(get_movie_earn,Element,Result), apply_get_movie_earn(get_movie_earn,Rest,RestResult)") #funzione ricorsiva Caso ricorsivo: entrambe le liste non sono vuote
    prolog.assertz("get_movies_from_director(DirectorName,Movies) :- findall(Id,movie(Id,Name,_,_,_,_,DirectorName,_,_,_,_,_,_,_,_),Movies)") #restituisce la lista di film di un regista
    prolog.assertz("get_earn_from_director(Director,Earn) :- get_movies_from_director(Director,Movies),apply_get_movie_earn(get_movie_earn,Movies,Movies_Earn),sumlist(Movies_Earn,Earn)") #restituisce i guadagni di un regista
    prolog.assertz("get_avg_earn_director(Director,Avgint):- get_movies_from_director(Director,Movies),length(Movies,Count),get_earn_from_director(Director,Earn),Avg is Earn/Count,round(Avg,Avgint)") #restituisce la media dei guadagni dei registi
    
    #conversione della lista in  lista di stringhe
    prolog.assertz("convert_list_to_string([],[])")#funzione ricorsiva Caso base: la lista è vuota
    prolog.assertz("convert_list_to_string([Atom|Rest],[String|RestStrings]):- atom_string(Atom,String),convert_list_to_string(Rest,RestStrings)") #funzione ricorsiva Caso ricorsivo: entrambe le liste non sono vuote
    
    #regole utili a ricavare la lista dei registi
    prolog.assertz("get_list_of_director(DirectorListS):- findall(Director,movie(Id,_,_,_,_,_,Director,_,_,_,_,_,_,_,_),DirectorList), convert_list_to_string(DirectorList,DirectorListS)") #restituisce la lista di registi con doppioni
    prolog.assertz("get_unique_directors(UniqueDirList) :- get_list_of_director(DirList),list_to_set(DirList, UniqueDirList)") #restituisce la lista di registi senza doppioni

    #regola per ricavare id dal regista
    prolog.assertz("get_id_from_director(Director,Id):-movie(Id,_,_,_,_,_,Director,_,_,_,_,_,_,_,_)")
    return prolog


#query sulla KB
def calculate_features(kb, final=False) -> dict:
    features_added = {}#definisco una variabile per contenere le feature da aggiungere
    bytes_list=list(kb.query(f"get_movies_category(Category)"))[0]["Category"] #creo una lista di categorie in base al valore del guadagno
    features_added["Category"]=list(map(lambda x: x.decode('utf-8'), bytes_list))# converto le categorie da atom a stringhe
    bytes_list=list(kb.query(f"get_unique_directors(DirectorList)"))[0]["DirectorList"]#genero una lista di tutti i registi
    features_added["Directors"]= list(map(lambda x: x.decode('utf-8'), bytes_list))#converto i registi da atom a stringhe
    lunghezza=len(features_added["Directors"])#ottengo il numero dei registi
    
    i=0
    Id=[]
    MoviesDirector=[]
    avgearndirector=[]
    movieearn=[]
    # questi sono i vari vettori di appoggio

    #ciclo per ottenere le liste che non abbiamo realizzato in prolog
    while(i<lunghezza):
        Director=features_added["Directors"][i]#variabile ausiliaria utile al fine di realizzare le query
        Id.append(list(kb.query(f"get_id_from_director('{Director}',Id)"))[0])#aggiunta di id dei director alla lista degli id
        MoviesDirector.append(list(kb.query(f"get_movies_from_director('{Director}',Movies)"))[0]["Movies"])#aggiungo elemento di un singolo regista 
        avgearndirector.append(list(kb.query(f"get_avg_earn_director('{Director}',Avg)"))[0]["Avg"])#aggiungo elemento earn corrispondente all'earn di quel regista
        i=i+1
    i=0
    
    while i < 5421:#ciclo per inserire il guadagno per ogni film
        movieearn.append(int(list(kb.query(f"get_movie_earn({i},Earn)"))[0]["Earn"]))#inserimento di guadagni per ogni film
        i=i+1
    print(np.mean(movieearn))

    features_added["IdDirectors"]=Id#assegno alla matrice di feature aggiunte i vettori d'appoggio
    features_added["MoviesDirector"]=MoviesDirector
    features_added["AvgearnDirector"]=avgearndirector
    features_added["MovieEarn"]=movieearn
    return features_added
    
    




def query_boolean_result(kb, query_str: str):
    return min(len(list(kb.query(query_str))), 1)

#creazione del dataset con la nuove feature
def produce_working_dataset(kb: Prolog, path: str, final=False):

    extracted_values_df = None

    working_dataset= pd.read_csv("dataset/working_dataset/movies.csv")
    features_added=calculate_features(kb)
    working_dataset["Earn"]=features_added["MovieEarn"]
    working_dataset["Category"]=features_added["Category"]
    working_dataset.to_csv(path, index=False, mode ="w")


def main():
    knowledge_base = create_kb()
    produce_working_dataset(knowledge_base, "dataset/generated_dataset/generated_dataset.csv")
    print("Created generated_dataset")

main()