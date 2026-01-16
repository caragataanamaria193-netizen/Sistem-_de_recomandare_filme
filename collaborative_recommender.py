import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import os

file_name = "movie_dataset.csv"

if not os.path.exists(file_name):
    file_name = "movie_dataset_final.csv"

try:
    df = pd.read_csv(file_name)
    print(f"✅ [Colaborativ] Succes: Am încărcat {file_name}")
except FileNotFoundError:
    print(f"❌ [Colaborativ] Eroare: Nu am găsit fișierul {file_name}!")
    exit()

movie_stats = df.groupby('title')['rating'].agg('count')

popular_movies_titles = movie_stats[movie_stats > 30].index
df_filtered = df[df['title'].isin(popular_movies_titles)].copy()

print(
    f"Statistici: Din {df['title'].nunique()} filme, am păstrat {df_filtered['title'].nunique()} pentru analiză (cele cu >30 voturi).")

movie_user_matrix = df_filtered.pivot_table(index='title', columns='user_id', values='rating').fillna(0)

movie_user_matrix_sparse = csr_matrix(movie_user_matrix.values)

model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
model_knn.fit(movie_user_matrix_sparse)

def recomanda_collaborative(titlu_film, top_n=10):
    """
    Recomandă filme bazat pe tiparul de votare al utilizatorilor.
    """
    if titlu_film not in movie_user_matrix.index:
        return [f"Filmul '{titlu_film}' nu are suficiente date pentru recomandări colaborative sau nu există."]

    query_index = movie_user_matrix.index.get_loc(titlu_film)

    distances, indices = model_knn.kneighbors(movie_user_matrix.iloc[query_index, :].values.reshape(1, -1),
                                              n_neighbors=top_n + 1)

    recomandari = []
    for i in range(1, len(distances.flatten())):
        idx = indices.flatten()[i]
        film_recomandat = movie_user_matrix.index[idx]
        recomandari.append(film_recomandat)

    return recomandari

if __name__ == "__main__":
    print("\n--- Sistem de Recomandare Colaborativ (User-Patterns) ---")

    test_movie = "Star Wars (1977)"  
    print(f"Căutăm recomandări pentru: {test_movie}")

    rezultate = recomanda_collaborative(test_movie)

    for i, film in enumerate(rezultate, 1):
        if "nu are suficiente date" in film:
            print(film)
        else:

            print(f"{i}. {film}")
