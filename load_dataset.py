import pandas as pd

path = "./ml-100k/"

ratings_cols = ["user_id", "movie_id", "rating", "timestamp"]
ratings = pd.read_csv(path + "u.data", sep="\t", names=ratings_cols)

movies_cols = ["movie_id", "title", "release_date", "video_release_date", "imdb_url", "unknown",
               "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary",
               "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
               "Sci-Fi", "Thriller", "War", "Western"]
movies = pd.read_csv(path + "u.item", sep="|", names=movies_cols, encoding="latin-1")

merged = ratings.merge(movies, on="movie_id")

merged.drop(columns=['timestamp', 'video_release_date', 'imdb_url'], inplace=True, errors='ignore')
merged.dropna(subset=['title', 'rating'], inplace=True)

movie_counts = merged.groupby('title')['rating'].count()
popular_movies = movie_counts[movie_counts >= 20].index
merged_filtered = merged[merged['title'].isin(popular_movies)]

user_movie_matrix = merged_filtered.pivot_table(index='user_id', columns='title', values='rating')

merged_filtered.to_csv("movie_dataset_final.csv", index=False)
user_movie_matrix.to_csv("user_movie_matrix.csv")

print("Statistici set de date:")
print(f"Total rating-uri: {ratings.shape[0]}")
print(f"Total filme: {movies.shape[0]}")
print(f"Randuri dupa filtrare (minim 20 voturi): {merged_filtered.shape[0]}")