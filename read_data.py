import sys, os
import pandas as pd
import numpy as np
import random 
from sklearn.datasets import load_svmlight_file

'''
movie_df: All ratings for all movies and info
rating_counts: Counts of ratings for each movie
avg_rating: Average rating for each movie
avg_count_df: Average rating and count of ratings for each movies
'''


class read_data:
    def __init__(self):
        self.test_fold = []

        self.valid_fold = []
        self.train_fold = []

    # Read in base data file
    def read_ml(self):
        data = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + "/ml-100k/u.data", sep="\t", header=None, names=[
                        "user_id", "movie_id", "rating", "timestamp"])

        # Read in movie info
        item = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + "/ml-100k/u.item", sep="|", header=None, names=[
                        "movie_id", "movie_title", "release_date", "video_release_date",
                        "imdb_url", "unknown", "Action", "Adventure", "Animation",
                        "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                        "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
                        "Thriller", "War", "Western"], encoding='ISO-8859-1', parse_dates=['release_date', 'video_release_date'])

        # Change timestamps to dates
        movie_df = data.join(item.set_index('movie_id'), on='movie_id')
        movie_df.timestamp = pd.to_datetime(movie_df.timestamp, unit='s')

        # Get average rating for each movie
        avg_rating_df = item[["movie_id"]].join(movie_df.groupby('movie_id').mean(
        ), on='movie_id').set_index("movie_id")[["rating"]].sort_values("rating", ascending=False)
        avg_rating_df.reset_index(level=0, inplace=True)

        # Get the number of ratings for each movie
        rating_counts_df = pd.DataFrame(movie_df.groupby('movie_id').count(
        )["user_id"])
        rating_counts_df["count"] = rating_counts_df.user_id
        rating_counts_df.drop('user_id', inplace=True, axis=1)
        rating_counts_df.sort_values('count', ascending=False, inplace=True)
        rating_counts_df.reset_index(level=0, inplace=True)

        # Joint average and counts
        avg_count_df = rating_counts_df.join(
            avg_rating_df.set_index('movie_id'), on='movie_id').sort_values(["count", "rating"], ascending=False)

        u1_base = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + "/ml-100k/u1.base", header=None, names=["user_id", "movie_id", "rating", "timestamp"], delimiter="\t")
        u2_base = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + "/ml-100k/u2.base", header=None, names=["user_id", "movie_id", "rating", "timestamp"], delimiter="\t")
        u3_base = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + "/ml-100k/u3.base", header=None, names=["user_id", "movie_id", "rating", "timestamp"], delimiter="\t")
        u4_base = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + "/ml-100k/u4.base", header=None, names=["user_id", "movie_id", "rating", "timestamp"], delimiter="\t")
        u5_base = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + "/ml-100k/u5.base", header=None, names=["user_id", "movie_id", "rating", "timestamp"], delimiter="\t")

        u1_test = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + "/ml-100k/u1.test", header=None, names=["user_id", "movie_id", "rating", "timestamp"], delimiter="\t")
        u2_test = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + "/ml-100k/u2.test", header=None, names=["user_id", "movie_id", "rating", "timestamp"], delimiter="\t")
        u3_test = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + "/ml-100k/u3.test", header=None, names=["user_id", "movie_id", "rating", "timestamp"], delimiter="\t")
        u4_test = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + "/ml-100k/u4.test", header=None, names=["user_id", "movie_id", "rating", "timestamp"], delimiter="\t")
        u5_test = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + "/ml-100k/u5.test", header=None, names=["user_id", "movie_id", "rating", "timestamp"], delimiter="\t")

        u1_base = u1_base.join(item.set_index("movie_id"), on="movie_id").drop(["timestamp", "release_date", "movie_title", "video_release_date", "imdb_url"], axis=1)
        u2_base = u2_base.join(item.set_index("movie_id"), on="movie_id").drop(["timestamp", "release_date", "movie_title", "video_release_date", "imdb_url"], axis=1)
        u3_base = u3_base.join(item.set_index("movie_id"), on="movie_id").drop(["timestamp", "release_date", "movie_title", "video_release_date", "imdb_url"], axis=1)
        u4_base = u4_base.join(item.set_index("movie_id"), on="movie_id").drop(["timestamp", "release_date", "movie_title", "video_release_date", "imdb_url"], axis=1)
        u5_base = u5_base.join(item.set_index("movie_id"), on="movie_id").drop(["timestamp", "release_date", "movie_title", "video_release_date", "imdb_url"], axis=1)

        u1_test = u1_test.join(item.set_index("movie_id"), on="movie_id").drop(["timestamp", "release_date", "movie_title", "video_release_date", "imdb_url"], axis=1)
        u2_test = u2_test.join(item.set_index("movie_id"), on="movie_id").drop(["timestamp", "release_date", "movie_title", "video_release_date", "imdb_url"], axis=1)
        u3_test = u3_test.join(item.set_index("movie_id"), on="movie_id").drop(["timestamp", "release_date", "movie_title", "video_release_date", "imdb_url"], axis=1)
        u4_test = u4_test.join(item.set_index("movie_id"), on="movie_id").drop(["timestamp", "release_date", "movie_title", "video_release_date", "imdb_url"], axis=1)
        u5_test = u5_test.join(item.set_index("movie_id"), on="movie_id").drop(["timestamp", "release_date", "movie_title", "video_release_date", "imdb_url"], axis=1)


        # First two columns are user_id and movie_id
        u1_base_np = u1_base.values
        u2_base_np = u2_base.values
        u3_base_np = u3_base.values
        u4_base_np = u4_base.values
        u5_base_np = u5_base.values
        base_list = [u1_base_np,u2_base_np,u3_base_np,u4_base_np,u5_base_np]
        u1_test_np = u1_test.values
        u2_test_np = u2_test.values
        u3_test_np = u3_test.values
        u4_test_np = u4_test.values
        u5_test_np = u5_test.values

        test_list = [u1_test_np,u2_test_np,u3_test_np,u4_test_np,u5_test_np]
        
        # qid = u1_base_np[:,0]
        # y = u1_base_np[:,2]
        # X = u1_base_np[:,3:]

        for i, base in enumerate(base_list):
            # random.shuffle(base)
            l_base = len(base)
            test_len = round(l_base * 0.8)
            # print(test_len,l_base)
            qid = base[:,0]
            y = base[:, 2]
            X = base[:, 3:]
            qid_vail = base[:,0]
            y_vail = base[:,2]
            X_vail = base[:,3:]
            qid_test = test_list[i][:,0]
            y_test = test_list[i][:,2]
            X_test = test_list[i][:,3:]
            self.train_fold.append((X,y,qid))
            self.valid_fold.append((X_vail,y_vail,qid_vail))
            self.test_fold.append((X_test,y_test,qid_test))
    
    def read_mq2008(self, path):
        for i in range(1,6):
            X, y, qid = load_svmlight_file('{}/Fold{}/train.txt'.format(path,i), query_id=True)
            X_test, y_test, qid_test = load_svmlight_file('{}/Fold{}/test.txt'.format(path,i), query_id=True)
            X_vali, y_vali, qid_vali = load_svmlight_file('{}/Fold{}/vali.txt'.format(path,i), query_id=True)
            X = X.toarray()
            X_test = X_test.toarray()
            X_vali = X_vali.toarray() 
            self.train_fold.append((X, y, qid))
            self.valid_fold.append((X_vali,y_vali,qid_vali))
            self.test_fold.append((X_test,y_test,qid_test))

    def get_fold(self, n):
        # print(len(self.train_fold))
        return self.train_fold[n], self.test_fold[n], self.valid_fold[n]