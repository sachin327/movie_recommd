from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import urllib.request

df = pd.read_csv('movie.csv', encoding= 'unicode_escape')
df['comb'] = df['actor_1_name'].str.lower() + ' ' + df['actor_2_name'].str.lower() + ' '+ df['actor_3_name'].str.lower() + ' '+ df['director_name'].str.lower() +' ' + df['genres'].str.lower()
df['comb'] = df['comb'].fillna('unknown')
df.set_index('movie_title', inplace = True)
df1=list(df['comb'])

# instantiating and generating the count matrix
count = CountVectorizer()
count_matrix = count.fit_transform(df['comb'])

# creating a Series for the movie titles so they are associated to an ordered numerical
# list I will use later to match the indexes
indices = pd.Series(df.index)
indices.head()

cosine_sim = cosine_similarity(count_matrix)
# function that takes in movie_title as input and returns the top 10 recommended movies

def recommendations(title):
    recommended_movies = []
    ratings = []
    actor_name=[]
    director_name=[]
    site=[]
    title=title.lower()
    try:
        # gettin the index of the movie that matches the title
        idx = indices[indices == title].index[0]

        # creating a Series with the similarity scores in descending order
        score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)

        # getting the indexes of the 10 most similar movies
        top_10_indexes = list(score_series.iloc[0:11].index)
 
        # populating the list with the titles of the best 10 matching movies
        for i in top_10_indexes:
            recommended_movies.append(list(df.index)[i])
            ratings.append(list(df['tmdb_ratings'])[i])
            actor_name.append(str(list(df['actor_1_name'])[i])[0].upper()+str(list(df['actor_1_name'])[i])[1:]+","+str(list(df['actor_2_name'])[i])[0].upper()+str(list(df['actor_2_name'])[i])[1:]+","+str(list(df['actor_3_name'])[i])[0].upper()+str(list(df['actor_3_name'])[i])[1:])
            director_name.append(list(df['director_name'])[i])
            site.append(list(df['movie_site'])[i])
        return recommended_movies, ratings,actor_name,director_name,site
    except Exception:
        title=title.lower()
        
        df2=np.array([title]+df1)
        count2 = CountVectorizer()
        count_matrix2 = count2.fit_transform(df2)
        c2=cosine_similarity(count_matrix2)
    
        score_series2 = pd.Series(c2[0]).sort_values(ascending = False)

        # getting the indexes of the 10 most similar movies
        top_10_indexes2 = list(score_series2.iloc[1:12].index)
    
        
        # populating the list with the titles of the best 10 matching movies
        for i in top_10_indexes2:
            recommended_movies.append(list(df.index)[i-1])
            ratings.append(list(df['tmdb_ratings'])[i-1])
            actor_name.append(str(list(df['actor_1_name'])[i-1])[0].upper()+str(list(df['actor_1_name'])[i-1])[1:].lower()+","+str(list(df['actor_2_name'])[i-1])[0].upper()+str(list(df['actor_2_name'])[i-1])[1:].lower()+","+str(list(df['actor_3_name'])[i-1])[0].upper()+str(list(df['actor_3_name'])[i-1])[1:].lower())
            director_name.append(list(df['director_name'])[i-1])
            site.append(list(df['movie_site'])[i-1])
        return recommended_movies, ratings,actor_name,director_name,site


        
app = Flask(__name__)

@app.route('/')
@app.route('/home') 
def home():
      return render_template('home.html')

@app.route("/recommend")
def recommend():
    title = request.args.get('title')
    rcmd=recommendations(title)
    m = rcmd[0]
    r = rcmd[1]
    a = rcmd[2]
    d = rcmd[3]
    s = rcmd[4]
    return render_template('recommend.html', title=title, m=m,r=r,a=a,d=d,s=s)

if __name__ == '__main__':
    app.run(debug=False)
