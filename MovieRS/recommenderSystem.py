import os
import pandas as pd

# configure file path
movies_path = './movies.csv'
ratings_path = './ratings.csv'

# read data
movies_dataframe = pd.read_csv(
    movies_path,
    usecols=['movieId', 'title'],
    dtype={'movieId': 'int32', 'title': 'str'})

ratings_dataframe = pd.read_csv(
    movies_path,
    usecols=['userId', 'movieId', 'rating'],
    dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

