import os
import sys
import pandas as pd
from operator import itemgetter
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from difflib import get_close_matches

def get_dataframes(movies_path, ratings_path):
    movies_dataframe = pd.read_csv(
        movies_path,
        usecols=['movieId', 'title'],
        dtype={'movieId': 'int32', 'title': 'str'})

    ratings_dataframe = pd.read_csv(
        ratings_path,
        usecols=['userId', 'movieId', 'rating'],
        dtype={'userId' : 'int32', 'movieId': 'int32', 'rating': 'float32'})
    return movies_dataframe, ratings_dataframe

def cleanup_ratings_df(ratings_df):
    movies_to_filter = pd.DataFrame(
        ratings_df.groupby('movieId').size(),
        columns=['count'])
    users_to_filter = pd.DataFrame(
        ratings_df.groupby('userId').size(),
        columns=['count'])
    popular_movies = list(set(movies_to_filter.query('count >= 6000').index))
    active_users = list(set(users_to_filter.query('count >= 0').index))
    movies_filter = ratings_df.movieId.isin(popular_movies).values
    users_filter = ratings_df.userId.isin(active_users).values
    ratings_filtered_df = ratings_df[movies_filter & users_filter]
    return ratings_filtered_df

def create_matrix_and_map(ratings_df, movies_df):
    movie_user_mat = ratings_df.pivot(
            index='movieId', columns='userId', values='rating').fillna(0)
    movies_map = {
        movie: i for i, movie in
        enumerate(list(movies_df.set_index('movieId').loc[movie_user_mat.index].title))
    }
    movie_user_mat = csr_matrix(movie_user_mat.values)
    return (movie_user_mat, movies_map)

def createKnnModel(movie_user_mat):
    knn_model = model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10, n_jobs=-1)

    knn_model.fit(movie_user_mat)
    return knn_model

def get_movie_index(movie_title, movies_map):
    titles = []
    indexes = []
    matches = []
    for title, idx in movies_map.items():
        titles.append(title)
        indexes.append(idx)
    matches = get_close_matches(movie_title, titles)
    if len(matches) == 0:
        print("No match found.")
        return -1
    print ("Result found in movie database : {0}".format(matches[0]))
    return indexes[titles.index(matches[0])]

def inp_loop(movie_user_mat, movies_map):
    index = -1
    while (index < 0):
        sys.stdout.write("Please type a movie title: ")
        sys.stdout.flush()
        inp = sys.stdin.readline()
        index = get_movie_index(inp, movies_map)
        movie = movie_user_mat[index]
    return movie

def do_prediction(knn_model, movie_user_mat, movies_map, movie):
    distances, indices = knn_model.kneighbors(movie, n_neighbors=11)
    recommends = \
            sorted(
                list(
                    zip(
                        indices.squeeze().tolist(),
                        distances.squeeze().tolist()
                    )
                ),
                key=itemgetter(1)
            )
    recommends.pop(0)
    recommends.reverse()
    return recommends

if __name__ == "__main__":
    print("Data processing...")
    movies_df, ratings_df = get_dataframes(
        movies_path='./ml-latest/movies.csv',
        ratings_path='./ml-latest/ratings.csv')
    ratings_df = cleanup_ratings_df(ratings_df)
    movie_user_mat, movies_map = create_matrix_and_map(ratings_df, movies_df)
    knn_model= createKnnModel(movie_user_mat)
    movie = inp_loop(movie_user_mat, movies_map)
    recommends = do_prediction(knn_model, movie_user_mat, movies_map, movie)

    reverse_map = {v: k for k, v in movies_map.items()}
    print('Recommendations :')
    for i, (indice, distance) in enumerate(recommends):
        print('{0}: {1}, with distance '
                'of {2}'.format(i+1, reverse_map[indice], distance))