import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.ensemble import RandomForestRegressor
import numpy as np

class MovieRecommender:
    def __init__(self, movies_df, user_data_df=None):
        self.df = movies_df
        self.user_data = user_data_df  
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['overview'])
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
        self.indices = pd.Series(self.df.index, index=self.df['title']).to_dict()
        
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self._train_collaborative_model()

    def _train_collaborative_model(self):
        if self.user_data is not None:
            
            movie_vectors = self.tfidf_matrix.toarray()
            movie_ratings = self.user_data.merge(self.df[['movie_id']], left_on='movie_id', right_index=True)
            X = movie_vectors[movie_ratings['movie_id']]
            y = movie_ratings['rating'].values
            
            self.rf_model.fit(X, y)

    def get_recommendations_by_title(self, title, use_rf_model=False):
        idx = self.indices.get(title)
        if idx is None:
            return []
        
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        movie_indices = [i[0] for i in sim_scores[1:11]]
        
        recommended_movies = self.df.iloc[movie_indices][['title', 'homepage']]
        if use_rf_model and self.user_data is not None:
            movie_vectors = self.tfidf_matrix[movie_indices].toarray()
            predicted_scores = self.rf_model.predict(movie_vectors)
            recommended_movies['predicted_score'] = predicted_scores
            recommended_movies = recommended_movies.sort_values(by='predicted_score', ascending=False)

        return recommended_movies

    def get_recommendations_by_description(self, description, use_rf_model=False):
        new_movie_vector = self.tfidf.transform([description])
        cosine_sim_new = linear_kernel(new_movie_vector, self.tfidf_matrix)
        sim_scores = list(enumerate(cosine_sim_new[0]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        movie_indices = [i[0] for i in sim_scores[:10]]
        recommended_movies = self.df.iloc[movie_indices][['title', 'homepage']]
        if use_rf_model and self.user_data is not None:
            movie_vectors = self.tfidf_matrix[movie_indices].toarray()
            predicted_scores = self.rf_model.predict(movie_vectors)
            recommended_movies['predicted_score'] = predicted_scores
            recommended_movies = recommended_movies.sort_values(by='predicted_score', ascending=False)
        
        return recommended_movies
