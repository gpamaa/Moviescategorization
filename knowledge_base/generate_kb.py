import pandas as pd
movies = pd.read_csv("dataset/working_dataset/movies.csv")
f = open("knowledge_base/movies.pl", "w+")
f.write("%movie(id, name, genre, year, score,votes,director,writer,star,country,budget,gross,company,runtime,month).\n")
for id in range(len(movies)):
   
    movie_info = movies.iloc[id]

    name = movie_info["name"]
    genre = movie_info["genre"]
    year = movie_info["year"]
    score = movie_info["score"]
    votes = movie_info["votes"]
    director = movie_info["director"]
    writer = movie_info["writer"]
    star = movie_info["star"]
    country = movie_info["country"]
    budget = movie_info["budget"]
    gross = movie_info["gross"]
    company = movie_info["company"]
    runtime = movie_info["runtime"]
    month = movie_info["month"]
    

    f.write(f"movie({id},\'{name}\',\'{genre}\', {year}, \'{score}\', {votes.astype(int)},\'{director}\', \'{writer}\', \'{star}\',\'{country}\',{budget.astype(int)},{gross},\'{company}\',{runtime.astype(int)},\'{month}\').\n")
   
    