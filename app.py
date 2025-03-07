from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
app = Flask(__name__)
df2 = pd.read_csv('C:\\jeevika\\tmdb_5000_movies.csv')  
df2['overview'] = df2['overview'].fillna('')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df2['overview'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df2.index, index=df2['title']).to_dict()

def get_recommendations_by_title(title):
    idx = indices.get(title)
    if idx is not None:
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        movie_indices = [i[0] for i in sim_scores[1:11]]  
        return df2[['title', 'homepage']].iloc[movie_indices]
    return pd.DataFrame(columns=['title', 'homepage']) 

def get_recommendations_by_description(description):
    new_movie_vector = tfidf.transform([description])
    cosine_sim_new = linear_kernel(new_movie_vector, tfidf_matrix)
    sim_scores = list(enumerate(cosine_sim_new[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores[1:11]]
    return df2[['title', 'homepage']].iloc[movie_indices]

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = pd.DataFrame(columns=['title', 'homepage']) 
    show_description = False 
    if request.method == 'POST':
        movie_title = request.form.get('movie_title', '').strip() 
        description = request.form.get('description', '').strip()  

        if movie_title in indices:
            recommendations = get_recommendations_by_title(movie_title)
        else:
            show_description = True  
            if description: 
                recommendations = get_recommendations_by_description(description)

    return render_template('index.html', recommendations=recommendations, show_description=show_description)

if __name__ == '__main__':
    app.run(debug=True)